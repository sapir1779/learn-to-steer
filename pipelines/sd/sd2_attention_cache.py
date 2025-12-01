import math
import torch

from pipelines.attention_cache import Cache, AttentionCache, CacheTypes


SD_BLOCK_TYPES = ["down", "mid", "up"]
SD_RESOLUTIONS = [8, 16, 32, 64]


class Sd2Cache(Cache):
    def __init__(self, block_types=SD_BLOCK_TYPES, store_heads=False):
        super().__init__(block_types, store_heads)

    def aggregate_values(self, output_res) -> torch.Tensor:
        """ Aggregates the values across the specified resolution """
        
        out = []
        for block_type in self.block_types:
            for attn_map in self.cache[block_type]:
                res = attn_map.shape[-1]
                if res == output_res:
                    out.append(attn_map)
                
        if len(out) > 0:
            if self.store_heads:
                out = torch.cat(out, dim=0).mean(dim=0)
            else:
                out = torch.stack(out, dim=0).mean(dim=0)
        
        return out


class Sd2AttentionCache(AttentionCache):
    def __init__(self, text_tokens, visual_tokens, map_size, num_inference_steps, target_timesteps=[], 
                 store_heads=False):
        super().__init__(text_tokens, visual_tokens, map_size, num_inference_steps, target_timesteps, 
                         store_heads)

    def init_base_cache(self):
        return Sd2Cache(SD_BLOCK_TYPES, self.store_heads)
        
    def _get_probs_value(self, cache_type, attn_probs):
        heads, HW, T = attn_probs.shape
        H = W = int(math.sqrt(HW))
        if cache_type == CacheTypes.CA_VISUAL_TEXT:
            # (heads, 4096, 77), for 64x64 attn maps
            value = attn_probs.reshape(heads, H, W, T).permute(0, 3, 1, 2)  ## (heads, text, height, width)
        else:
            # Not implemented
            value = None
        
        return value
        
    def store_attention_map(self, block_type, attn_probs, is_cross, batch_size):
        if not is_cross or not self.is_initialized():
            return
        
        if batch_size > 1:
            prompt_attn_start_idx = attn_probs.shape[0] // 2
            attn_probs = attn_probs[prompt_attn_start_idx:]
            
        for cache_type in self.cache_types:
            value = self._get_probs_value(cache_type, attn_probs)
            self.add(cache_type, block_type, value)
