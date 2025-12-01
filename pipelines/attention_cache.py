from collections import defaultdict
from enum import Enum
from abc import ABC, abstractmethod
import torch


class Cache:
    """ A basic cache for attention maps """
    def __init__(self, block_types, store_heads=False):
        self.block_types = block_types
        self.store_heads = store_heads
        self.cache = {block_type: [] for block_type in block_types}
        
    def add(self, block_type, value):
        if not self.store_heads:
            value = value.mean(0)
        
        self.cache[block_type].append(value)
        
    def get_attention_maps(self, block_type, layer_idx):
        return self.cache[block_type][layer_idx]
    
    def get_all_attn_maps(self):
        out = []
        for block_type in self.block_types:
            block_cache = self.cache[block_type]
            for idx in range(len(block_cache)):
                out.append(block_cache[idx])
                
        if len(out) > 0:
            # List[(24, 256, 256)] --> (57, 24, 256, 256)
            out = torch.stack(out, dim=0)
        return out
    

    def get_attn_maps_from_indices(self, token_inds):
        if len(token_inds) == 0:
            return None
        
        start_idx = token_inds[0]
        end_idx = start_idx + len(token_inds)
        attn_maps_per_resolution = defaultdict(list)
        for block_type in self.block_types:
            for attn_map in self.cache[block_type]:
                # (heads, text, height, width) --> (heads, N, height, width), where N = num of tokens
                out_map = attn_map[:, start_idx: end_idx, :, :]
                res = out_map.shape[-1]
                attn_maps_per_resolution[res].append(out_map)
        
        output_attn_maps = {}
        for res in attn_maps_per_resolution.keys():
            output_attn_maps[res] = torch.stack(attn_maps_per_resolution[res], dim=0)
            
        return output_attn_maps


class CacheTypes(Enum):
    CA_VISUAL_TEXT = 0
    SA_VISUAL = 1
    CA_TEXT_VISUAL = 2
    SA_TEXT = 3


class AttentionCache(ABC):
    """
    Represents a cache for all types of attention maps (self- and cross-attention).
    Responsible for caching the attention maps for each denoising timestep.
    """
    def __init__(self, text_tokens, visual_tokens, map_size, num_inference_steps,
                 target_timesteps=[], store_heads=False):
        self.text_tokens = text_tokens
        self.visual_tokens = visual_tokens
        self.map_size = map_size
        self.num_inference_steps = num_inference_steps
        self.target_timesteps = target_timesteps
        self.store_heads = store_heads
        self.current_timestep = -1
        
        self.cache_types = [CacheTypes.CA_VISUAL_TEXT]
        
        self.init_caches()
        
    def is_initialized(self):
        return self.current_timestep >= 0
    
    @abstractmethod
    def init_base_cache(self) -> Cache: # type: ignore
        """ Each inheriting class should implement this. """
        pass
        
    def init_caches(self):
        self.caches = {}
        for cache_type in self.cache_types:
            self.caches[cache_type] = [None] * self.num_inference_steps
        
    def init_timestep(self, timestep):
        self.current_timestep = timestep
        self.reset_current_timestep()
    
    def handle_timestep_end(self):
        if not self.is_initialized():
            return
        
        # Delete the current caches if it's unnecessary to keep them for the next timestep
        if len(self.target_timesteps) == 0 or (self.current_timestep not in self.target_timesteps):
            for cache_list in self.caches.values():
                cache_list[self.current_timestep] = None
                
    def reset_current_timestep(self):
        if not self.is_initialized():
            return
        
        for cache_list in self.caches.values():
            cache_list[self.current_timestep] = self.init_base_cache()
            
    def get_cache(self, cache_type):
        return self.caches[cache_type]
    
    def get_cross_attn_visual_text_cache(self):
        return self.caches[CacheTypes.CA_VISUAL_TEXT] if CacheTypes.CA_VISUAL_TEXT in self.caches else None
    
    def add(self, cache_type, block_type, value):
        if not self.is_initialized():
            return
        
        self.caches[cache_type][self.current_timestep].add(block_type, value)
