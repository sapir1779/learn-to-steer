# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch

from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.utils import (
    is_torch_xla_available,
    logging,
)

from pipelines.flux.rf_inversion.pipeline_flux_rf_inversion import (
    RFInversionFluxPipeline,
    retrieve_timesteps, 
    calculate_shift,
)
from pipelines.flux.flux_attention_processor import FluxAttnProcessorDebug2_0
from pipelines.flux.flux_attention_cache import FluxAttentionCache
from pipelines.config import Config
from utils import process_pipe_config

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class RFInversionFluxPipelineDebug(RFInversionFluxPipeline):
    def register_attention_processors(self):
        if not self.conf.attention.enable:
            return
        
        attention_processors = {}
        for name in self.transformer.attn_processors.keys():
            attention_processors[name] = FluxAttnProcessorDebug2_0(self.attention_cache, name) 
        
        self.original_attention_processors = self.transformer.attn_processors
        self.transformer.set_attn_processor(attention_processors)
        
    def reinstate_original_attention_processors(self):
        if self.conf.attention.enable:
            self.transformer.set_attn_processor(self.original_attention_processors)
        

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        inverted_latents: Optional[torch.FloatTensor] = None,
        image_latents: Optional[torch.FloatTensor] = None,
        latent_image_ids: Optional[torch.FloatTensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 1.0,
        decay_eta: Optional[bool] = False,
        eta_decay_power: Optional[float] = 1.0,
        strength: float = 1.0,
        start_timestep: float = 0,
        stop_timestep: float = 0.25,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        input_config: Config = None,
        relation_classifier = None,
        prompt_parser = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            inverted_latents (`torch.Tensor`, *optional*):
                The inverted latents from `pipe.invert`.
            image_latents (`torch.Tensor`, *optional*):
                The image latents from `pipe.invert`.
            latent_image_ids (`torch.Tensor`, *optional*):
                The latent image ids from `pipe.invert`.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            eta (`float`, *optional*, defaults to 1.0):
                The controller guidance, balancing faithfulness & editability:
                higher eta - better faithfullness, less editability. For more significant edits, lower the value of eta.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        self.prompt = prompt
        self.num_inference_steps = num_inference_steps
        self.seed = generator.initial_seed() if generator is not None else -1
        self.img_width = width
        self.img_height = height
        self.max_sequence_length = max_sequence_length
        process_pipe_config(self, input_config)
        
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            inverted_latents,
            image_latents,
            latent_image_ids,
            height,
            width,
            start_timestep,
            stop_timestep,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        do_rf_inversion = inverted_latents is not None

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        if do_rf_inversion:
            latents = inverted_latents
        else:
            latents, latent_image_ids = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = (int(height) // self.vae_scale_factor // 2) * (int(width) // self.vae_scale_factor // 2)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        if do_rf_inversion:
            start_timestep = int(start_timestep * num_inference_steps)
            stop_timestep = min(int(stop_timestep * num_inference_steps), num_inference_steps)
            timesteps, sigmas, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None
            
        # Register attention processors for storing the attention activations later on
        self.attention_cache = FluxAttentionCache(
            max_sequence_length, 
            height, 
            self.conf.attention.map_size,
            num_inference_steps,
            self.conf.attention.target_timesteps,
            self.conf.attention.store_heads,
        )
        self.register_attention_processors()
        
        all_latents = []
        if do_rf_inversion:
            y_0 = image_latents.clone()
        # 6. Denoising loop / Controlled Reverse ODE, Algorithm 2 from: https://arxiv.org/pdf/2410.10792
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if do_rf_inversion:
                    # ti (current timestep) as annotated in algorithm 2 - i/num_inference_steps.
                    t_i = 1 - t / 1000
                    # dt = torch.tensor(1 / (len(timesteps) - 1), device=device)

                if self.interrupt:
                    continue
                
                self.attention_cache.init_timestep(i)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                
                self.attention_cache.reset_current_timestep()
                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                latents_dtype = latents.dtype
                if do_rf_inversion:
                    v_t = -noise_pred
                    v_t_cond = (y_0 - latents) / (1 - t_i)
                    eta_t = eta if start_timestep <= i < stop_timestep else 0.0
                    if decay_eta:
                        eta_t = eta_t * (1 - i / num_inference_steps) ** eta_decay_power  # Decay eta over the loop
                    v_hat_t = v_t + eta_t * (v_t_cond - v_t)

                    # SDE Eq: 17 from https://arxiv.org/pdf/2410.10792
                    latents = latents + v_hat_t * (sigmas[i] - sigmas[i + 1])
                else:
                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                self.attention_cache.handle_timestep_end()
                
                if self.conf.decode_all_latents:
                    all_latents.append(latents.clone().detach())
                
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents
        elif self.conf.decode_all_latents:
            def decode_latents(latents_to_decode):
                latents_to_decode = self._unpack_latents(latents_to_decode, height, width, self.vae_scale_factor)
                latents_to_decode = (latents_to_decode / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                image = self.vae.decode(latents_to_decode, return_dict=False)[0]
                image = self.image_processor.postprocess(image, output_type=output_type)
                return image
            
            images = [decode_latents(latents_to_decode) for latents_to_decode in all_latents]
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()
        
        # Reinstate the original transformer's attention processors
        self.reinstate_original_attention_processors()
        
        if not return_dict:
            return (image,)
        elif self.conf.decode_all_latents:
            return FluxPipelineOutput(images=images)

        return FluxPipelineOutput(images=image)
