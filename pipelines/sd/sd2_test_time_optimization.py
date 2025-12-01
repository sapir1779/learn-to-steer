
import numpy as np
import torch

from utils import extract_cross_attn_maps
from pipelines.test_time_optimization import TestTimeOptimization

torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True, precision=4)


class Sd2TestTimeOptimization(TestTimeOptimization):
    def __init__(self, num_inference_steps, prompt_parser, relation_classifier, test_time_opt_config, attn_cache, 
                 unet, tokenizer, 
                 timestep_cond, 
                 cross_attention_kwargs,
                 added_cond_kwargs,
                 prompt_embeds,
                 prompt, workspace_path, device):
        
        super().__init__(num_inference_steps, prompt_parser, relation_classifier, test_time_opt_config, 
                         attn_cache, tokenizer, 
                         prompt, workspace_path, device)
        self.unet = unet
        self.timestep_cond = timestep_cond
        self.cross_attention_kwargs = cross_attention_kwargs
        self.added_cond_kwargs = added_cond_kwargs
        self.prompt_embeds = prompt_embeds

    def denoise(self, latents, timestep): # type: ignore
        self.attn_cache.reset_current_timestep()
        latents = latents.clone().detach().requires_grad_(True)
        noise_pred = self.unet(
            latents,
            timestep,
            encoder_hidden_states=self.prompt_embeds,
            timestep_cond=self.timestep_cond,
            cross_attention_kwargs=self.cross_attention_kwargs,
            added_cond_kwargs=self.added_cond_kwargs,
            return_dict=False,
        )[0]
        return latents, noise_pred
    
    def extract_cross_attn_maps(self, obj1, obj2, rel_name, iter_idx): # type: ignore
        return extract_cross_attn_maps(self.tokenizer, self.attn_cache, self.prompt, 
                                        obj1, obj2, rel_name, iter_idx, add_batch_dim=False, 
                                        output_all_heads_and_layers=True, flatten_maps=True, 
                                        remove_start_token=True)
        