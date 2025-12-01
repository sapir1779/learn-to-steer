import torch

from pipelines.attention_cache import Cache, AttentionCache, CacheTypes


FLUX_BLOCK_TYPES = ["double", "single"]


class FluxCache(Cache):
    def __init__(self, block_types=FLUX_BLOCK_TYPES, store_heads=False):
        super().__init__(block_types, store_heads)

    def aggregate_values(self, block_types=FLUX_BLOCK_TYPES, layer_idx=None) -> torch.Tensor:
        """ Aggregates the values across the specified Transformer block types """
        
        if type(block_types) is not list:
            block_types = [block_types]
        
        if layer_idx is not None and len(block_types) > 1:
            print("Not part of the API")
            return []
            
        out = []
        for block_type in block_types:
            block_cache = self.cache[block_type]
            
            if (layer_idx is not None) and (0 <= layer_idx < len(block_cache)):
                out.append(block_cache[layer_idx])
            else:
                for idx in range(len(block_cache)):
                    out.append(block_cache[idx])
                
        if len(out) > 0:
            if self.store_heads:
                out = torch.cat(out, dim=0).mean(dim=0)
            else:
                out = torch.stack(out, dim=0).mean(dim=0)
        
        return out


class FluxAttentionCache(AttentionCache):
    def __init__(self, text_tokens, visual_tokens, map_size, num_inference_steps, target_timesteps=[], 
                 store_heads=False):
        super().__init__(text_tokens, visual_tokens, map_size, num_inference_steps, target_timesteps, 
                         store_heads)

    def init_base_cache(self):
        return FluxCache(FLUX_BLOCK_TYPES, self.store_heads)
        
    def _get_probs_value(self, cache_type, joint_logits):
        # Note: we're not computing softmax on the joint text-visual columns, 
        # but on separate columns per type: {text, visual}
        
        T = self.text_tokens
        H = W = self.map_size
        heads = joint_logits.shape[0]
        if cache_type == CacheTypes.CA_VISUAL_TEXT:
            # (heads, 256, 512)
            logits = joint_logits[:, T:, :T]
            value = torch.softmax(logits, dim=-1)
            value = value.reshape(heads, H, W, T).permute(0, 3, 1, 2)  ## (heads, text, height, width)
        elif cache_type == CacheTypes.CA_TEXT_VISUAL:
            # (heads, 512, 256)
            logits = joint_logits[:, :T, T:]
            value = torch.softmax(logits, dim=1)
            value = value.reshape(heads, T, H, W)  ## (heads, text, height, width)
        elif cache_type == CacheTypes.SA_TEXT:
            logits = joint_logits[:, :T, :T]
            value = torch.softmax(logits, dim=-1)
        elif cache_type == CacheTypes.SA_VISUAL:
            logits = joint_logits[:, T:, T:]
            value = torch.softmax(logits, dim=-1)
        
        return value
        
    def store_attention_map(self, block_type, joint_logits):
        if not self.is_initialized():
            return
        
        joint_logits = joint_logits.squeeze()
        for cache_type in self.cache_types:
            value = self._get_probs_value(cache_type, joint_logits)
            self.add(cache_type, block_type, value)
