from typing import Any, Callable, Dict, List, Optional, Union
import torch
from typing import List, Optional, Union
from PIL import Image
import torch
from easydict import EasyDict

from diffusers.utils import (
    is_torch_xla_available,
    logging,
)

from pipelines.sd.FixedPointInversion.inversion import MyInvertFixedPoint
from pipelines.sd.sd2_attention_processor import Sd2AttnProcessorDebug2_0
from pipelines.sd.sd2_attention_cache import Sd2AttentionCache
from pipelines.config import Config
from utils import process_pipe_config


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def center_crop(im):
    width, height = im.size  # Get dimensions
    min_dim = min(width, height)
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im

def load_im_into_format_from_img(img):
    return center_crop(img).resize((512, 512))

def load_im_into_format_from_path(im_path):
    return load_im_into_format_from_img(Image.open(im_path))


class Sd2InversionPipelineDebug(MyInvertFixedPoint):
    def __init__(self, vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor,
                 requires_safety_checker: bool = True):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor,
                         requires_safety_checker)
    
    def register_attention_processors(self):
        if not self.conf or not self.conf.attention.enable:
            return
        
        attention_processors = {}
        for name in self.unet.attn_processors.keys():
            attention_processors[name] = Sd2AttnProcessorDebug2_0(self.attention_cache, name)

        self.original_attention_processors = self.unet.attn_processors
        self.unet.set_attn_processor(attention_processors)
            
    def reinstate_original_attention_processors(self):
        if self.conf and self.conf.attention.enable:
            self.unet.set_attn_processor(self.original_attention_processors)
            
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        input_config: Config = None,
        relation_classifier = None,
        start_timestep: int = 0,
    ):
        self.prompt = prompt[0] if type(prompt) is list else prompt
        self.num_inference_steps = num_inference_steps
        self.seed = -1
        self.img_width = width
        self.img_height = height
        self.max_sequence_length = 77
        process_pipe_config(self, input_config)
        
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            None,
            latents,
        )
        
        # Register attention processors for storing the attention activations later on
        if self.conf:
            map_size = self.conf.attention.map_size
            target_timesteps = self.conf.attention.target_timesteps
            store_heads = self.conf.attention.store_heads
        else:
            map_size = 16
            target_timesteps = []
            store_heads = True
        
        self.attention_cache = Sd2AttentionCache(
            77, 
            height, 
            map_size, 
            num_inference_steps,
            target_timesteps,
            store_heads,
        )
        self.register_attention_processors()
        
        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                self.attention_cache.init_timestep(i)
                
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                # predict the noise residual
                self.attention_cache.reset_current_timestep()
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)[0]
                
                self.attention_cache.handle_timestep_end()
                
                progress_bar.update()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor)[0]
        else:
            image = latents

        image = self.image_processor.postprocess(image, output_type=output_type)
        
        # Reinstate the original transformer's attention processors
        self.reinstate_original_attention_processors()

        if not return_dict:
            return image

        if output_type == 'pil':
            return EasyDict({'images': image, 'latents': latents})
        else:
            EasyDict({'images': image})
