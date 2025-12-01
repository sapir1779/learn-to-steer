import os
import torch

from pipelines.sd.sd2_attention_cache import SD_RESOLUTIONS
from pipelines.drawing_utils import (
    create_folders,
    concat_images,
    MapsDrawer,
    MapsDrawerNoCache,
)


class Sd2MapsDrawer(MapsDrawer):
    def __init__(self, config, skip_prefix_tokens=False, objects=[]):
        super().__init__(config, skip_prefix_tokens, objects)
        
    def draw_maps_per_resolution(self, cache_obj, prompt, tokenizer, bg_image, output_folder):
        """ Draws the cross-attention maps averaged over each block (double, single) """
        if cache_obj is None:
            return
        
        _, folder_grey, folder_color = create_folders(self.include_color, output_folder, "per_resolution")
        
        # Per block, averaged over all resolutions
        res_names = ""
        images_grey = []
        images_color = []
        for res in SD_RESOLUTIONS:
            attn_maps = cache_obj.aggregate_values(res)
            if len(attn_maps) == 0:
                continue
            
            images_grey, images_color = self.draw_and_append_maps(attn_maps, prompt, tokenizer, bg_image,
                                                                  images_grey, images_color)
            res_names = f"{res_names}__{res}"
            
        filename = f"CA__{res_names}.png"
        concat_images(images_grey, display_img=False, output_path=os.path.join(folder_grey, filename))
        concat_images(images_color, display_img=False, output_path=os.path.join(folder_color, filename))


    def dump_attention_maps_to_workspace(self, cache_obj, prompt, tokenizer, bg_image, output_folder, step_idx=None):
        if step_idx is not None:
            output_folder = os.path.join(output_folder, f"step_{step_idx}")
        
        if 'resolution' in self.dump_sources:
            self.draw_maps_per_resolution(cache_obj, prompt, tokenizer, bg_image, output_folder)


    def dump_attention_maps_to_workspace_per_step(self, attn_cache, prompt, tokenizer, bg_image):
        attn_maps_cache = attn_cache.get_cross_attn_visual_text_cache()
        output_folder = os.path.join(self.config.workspace_path, "CA_VISUAL_TEXT")
        for step_idx in attn_cache.target_timesteps:
            cache_obj = attn_maps_cache[step_idx]
            self.dump_attention_maps_to_workspace(cache_obj, prompt, tokenizer, bg_image, output_folder, step_idx)
            
            
class Sd2MapsDrawerNoCache(MapsDrawerNoCache):
    def __init__(self, add_row_indices=True, res16=False, avg=False, map_size=16):
        super().__init__(add_row_indices, res16, avg, map_size)
        
    def draw_maps_per_resolution(self, objects_attn_maps, rel_objects, bg_image, output_folder, per_block=False):
        """ 
        Draws the cross-attention maps in each layer of each block ("double", "single"), 
        where in each layer we compute the mean over the heads, and we draw the layers of a block in a single image.
        """
        
        _, folder_grey, folder_color = create_folders(self.include_color, output_folder, "per_resolution")
        
        block_attn_maps_dict = {}
        total_flat_maps = objects_attn_maps[0].shape[0]
        if total_flat_maps == 128:  ## SD 1.4
            dict_layers_heads = {
                8: (1, 8),  # res 8x8, 1 layer 8 heads per layer
                16: (5, 8),  # res 16x16, 5 layer 8 heads per layer
                32: (5, 8),  # res 32x32, 5 layer 8 heads per layer
                64: (5, 8),  # res 64x64, 5 layer 8 heads per layer
            }
            
        elif total_flat_maps == 40:  ## SD 1.4 only 16x16
            dict_layers_heads = {
                16: (5, 8),  # res 16x16, 5 layer 8 heads per layer
            }
        else:
            dict_layers_heads = {}
        
        prev_start_idx = 0
        for res in dict_layers_heads.keys():
            layers, heads = dict_layers_heads[res]
            N = layers * heads
            attn_maps = []
            for i in range(len(objects_attn_maps)):
                res_attn_maps = objects_attn_maps[i][prev_start_idx: prev_start_idx+N]
                attn_maps.append(res_attn_maps)
            
            prev_start_idx += N
            attn_maps = torch.stack(attn_maps, dim=0)
            block_attn_maps_dict[res] = attn_maps
            
        images_grey = []
        images_color = []
        res_names = ""
        for res, attn_maps_tensor in block_attn_maps_dict.items():
            if self.res16 and res != 16:
                continue
            
            attn_maps = attn_maps_tensor.mean(dim=1)
            images_grey, images_color = self.draw_and_append_maps(attn_maps, rel_objects, bg_image, 
                                                                    images_grey, images_color)
            res_names = f"{res_names}__{res}"
        
        filename = f"CA__{res_names}.png"
        concat_images(images_grey, display_img=False, output_path=os.path.join(folder_grey, filename), 
                      add_row_indices=self.add_row_indices)
        concat_images(images_color, display_img=False, output_path=os.path.join(folder_color, filename),
                      add_row_indices=self.add_row_indices)
        
  
    def draw_maps_per_layer(self, objects_attn_maps, rel_objects, bg_image, output_folder, per_block=False):
        """ 
        Draws the cross-attention maps in each layer of each block ("double", "single"), 
        where in each layer we compute the mean over the heads, and we draw the layers of a block in a single image.
        """
        
        _, folder_grey, folder_color = create_folders(self.include_color, output_folder, "per_layer")
        
        block_attn_maps_dict = {}
        total_flat_maps = objects_attn_maps[0].shape[0]
        if total_flat_maps == 128:  ## SD 1.4
            dict_layers_heads = {
                0: (1, 8),  # res 8x8, 1 layer 8 heads per layer
                1: (5, 8),  # res 16x16, 5 layer 8 heads per layer
                2: (5, 8),  # res 32x32, 5 layer 8 heads per layer
                3: (5, 8),  # res 64x64, 5 layer 8 heads per layer
            }
            
        elif total_flat_maps == 40:  ## SD 1.4 only 16x16
            dict_layers_heads = {
                0: (5, 8),  # res 16x16, 5 layer 8 heads per layer
            }
        else:
            dict_layers_heads = {}
        
        prev_start_idx = 0
        for j in range(len(dict_layers_heads.keys())):
            layers, heads = dict_layers_heads[j]
            N = layers * heads
            attn_maps = []
            for i in range(len(objects_attn_maps)):
                res_attn_maps = objects_attn_maps[i][prev_start_idx: prev_start_idx+N]
                a = res_attn_maps.reshape(layers, heads, 16, 16)
                attn_maps.append(a)
            
            prev_start_idx += N
            attn_maps = torch.stack(attn_maps, dim=0)
            block_attn_maps_dict[j] = attn_maps
            
        images_grey = []
        images_color = []
        for block_type, attn_maps_tensor in block_attn_maps_dict.items():
            for layer_idx in range(attn_maps_tensor.shape[1]):
                attn_maps = attn_maps_tensor[:, layer_idx].mean(dim=1)
                images_grey, images_color = self.draw_and_append_maps(attn_maps, rel_objects, bg_image, 
                                                                        images_grey, images_color)
        
        filename = f"CA__layers.png"
        concat_images(images_grey, display_img=False, output_path=os.path.join(folder_grey, filename), 
                        num_cols=4, num_rows=4, add_row_indices=self.add_row_indices)
        concat_images(images_color, display_img=False, output_path=os.path.join(folder_color, filename), 
                        num_cols=4, num_rows=4, add_row_indices=self.add_row_indices)


    def dump_attention_maps_to_workspace(self, objects_attn_maps, rel_objects, bg_image, output_folder):
        # self.draw_maps_overall_mean(objects_attn_maps, rel_objects, bg_image, output_folder)
        self.draw_maps_per_resolution(objects_attn_maps, rel_objects, bg_image, output_folder)
