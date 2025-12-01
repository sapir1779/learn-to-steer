
import numpy as np
import torch

from utils import extract_cross_attn_maps
from pipelines.test_time_optimization import TestTimeOptimization

torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True, precision=4)


class FluxTestTimeOptimization(TestTimeOptimization):
    def __init__(self, num_inference_steps, prompt_parser, relation_classifier, test_time_opt_config, attn_cache, 
                 transformer, tokenizer, joint_attention_kwargs, guidance, pooled_prompt_embeds, 
                 prompt_embeds, text_ids, latent_image_ids, prompt, workspace_path, device):
        
        super().__init__(num_inference_steps, prompt_parser, relation_classifier, test_time_opt_config, 
                         attn_cache, tokenizer, 
                         prompt, workspace_path, device)
        self.transformer = transformer
        self.joint_attention_kwargs = joint_attention_kwargs
        self.guidance = guidance
        self.pooled_prompt_embeds = pooled_prompt_embeds
        self.prompt_embeds = prompt_embeds
        self.text_ids = text_ids
        self.latent_image_ids = latent_image_ids

    def denoise(self, latents, timestep): # type: ignore
        self.attn_cache.reset_current_timestep()
        latents = latents.clone().detach().requires_grad_(True)
        noise_pred = self.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=self.guidance,
            pooled_projections=self.pooled_prompt_embeds,
            encoder_hidden_states=self.prompt_embeds,
            txt_ids=self.text_ids,
            img_ids=self.latent_image_ids,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False,
        )[0]
        return latents, noise_pred
    
    def extract_cross_attn_maps(self, obj1, obj2, rel_name, iter_idx): # type: ignore
        return extract_cross_attn_maps(self.tokenizer, self.attn_cache, self.prompt, 
                                        obj1, obj2, rel_name, iter_idx, add_batch_dim=False, 
                                        output_all_heads_and_layers=True, flatten_maps=True,
                                        target_map_size=self.target_map_size)
        