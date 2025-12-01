import os
import torch

from pipelines.attention_cache import CacheTypes
from pipelines.flux.flux_attention_cache import FLUX_BLOCK_TYPES
from pipelines.drawing_utils import (
    create_folders,
    concat_images,
    MapsDrawer,
    MapsDrawerNoCache,
)


class FluxMapsDrawer(MapsDrawer):
    def __init__(self, config, skip_prefix_tokens=False, objects=[]):
        super().__init__(config, skip_prefix_tokens, objects)

    def draw_maps_overall_mean_all_steps(self, attn_maps_cache, target_timesteps, 
                                         prompt, tokenizer, bg_image, output_folder):
        """ 
        Draws the cross-attention maps averaged over every image resolution and block type, 
        across all of the steps in a single image 
        """
        output_path = create_folders(self.include_color, output_folder, "overall", create_sub_folders=False)
        images_grey, images_color = [], []
        for step_idx in target_timesteps:
            cache_obj = attn_maps_cache[step_idx]
            if cache_obj is None:
                continue
            
            attn_maps = cache_obj.aggregate_values()
            images_grey, images_color = self.draw_and_append_maps(attn_maps, prompt, tokenizer, bg_image,
                                                                  images_grey, images_color)
            
        filename = f"CA.png"
        concat_images(images_grey, display_img=False, output_path=os.path.join(output_path, f"grey__{filename}"))
        concat_images(images_color, display_img=False, output_path=os.path.join(output_path, f"color__{filename}"))
    
    def draw_maps_overall_mean(self, cache_obj, prompt, tokenizer, bg_image, output_folder):
        """ Draws the cross-attention maps averaged over every image resolution and block type """
        if cache_obj is None:
            return
        
        output_path = create_folders(self.include_color, output_folder, "overall", create_sub_folders=False)
        attn_maps = cache_obj.aggregate_values()
        images_grey, images_color = self.draw_and_append_maps(attn_maps, prompt, tokenizer, bg_image)
        filename = f"CA.png"
        concat_images(images_grey, display_img=False, output_path=os.path.join(output_path, f"grey__{filename}"))
        concat_images(images_color, display_img=False, output_path=os.path.join(output_path, f"color__{filename}"))


    def draw_maps_per_block(self, cache_obj, prompt, tokenizer, bg_image, output_folder):
        """ Draws the cross-attention maps averaged over each block (double, single) """
        if cache_obj is None:
            return
        
        _, folder_grey, folder_color = create_folders(self.include_color, output_folder, "per_block")
        
        # Per block, averaged over all resolutions
        block_names = ""
        images_grey = []
        images_color = []
        for block_type in FLUX_BLOCK_TYPES:
            attn_maps = cache_obj.aggregate_values(block_type)
            if len(attn_maps) == 0:
                continue
            
            images_grey, images_color = self.draw_and_append_maps(attn_maps, prompt, tokenizer, bg_image,
                                                                  images_grey, images_color)
            block_names = f"{block_names}__{block_type.upper()}"
            
        filename = f"CA__{block_names}.png"
        concat_images(images_grey, display_img=False, output_path=os.path.join(folder_grey, filename))
        concat_images(images_color, display_img=False, output_path=os.path.join(folder_color, filename))

    def draw_maps_per_head(self, cache_obj, prompt, tokenizer, bg_image, output_folder):
        """ Draws the heads of the cross-attention maps in each layer """
        if cache_obj is None:
            return
        
        _, folder_grey, folder_color = create_folders(self.include_color, output_folder, "per_head")
        for block_type, attn_maps_list in cache_obj.cache.items():
            for layer_idx in range(len(attn_maps_list)):
                images_grey = []
                images_color = []
                attn_maps_per_head = cache_obj.get_attention_maps(block_type, layer_idx)
                for h in range(attn_maps_per_head.shape[0]):
                    attn_maps = attn_maps_per_head[h]
                    images_grey, images_color = self.draw_and_append_maps(attn_maps, prompt, tokenizer, bg_image, 
                                                                            images_grey, images_color)

                filename = f"CA__{block_type.upper()}__L_{layer_idx}.png"
                concat_images(images_grey, display_img=False, output_path=os.path.join(folder_grey, filename))
                concat_images(images_color, display_img=False, output_path=os.path.join(folder_color, filename))

    def draw_maps_per_layer(self, cache_obj, prompt, tokenizer, bg_image, output_folder, per_block=False):
        """ 
        Draws the cross-attention maps in each layer of each block ("double", "single"), 
        where in each layer we compute the mean over the heads, and we draw the layers of a block in a single image.
        """
        if cache_obj is None:
            return
        
        _, folder_grey, folder_color = create_folders(self.include_color, output_folder, "per_layer")
        if per_block:
            for block_type, attn_maps_list in cache_obj.cache.items():
                images_grey = []
                images_color = []
                for layer_idx in range(len(attn_maps_list)):
                    attn_maps = cache_obj.aggregate_values(block_type, layer_idx)
                    images_grey, images_color = self.draw_and_append_maps(attn_maps, prompt, tokenizer, bg_image, 
                                                                            images_grey, images_color)
                
                filename = f"CA__{block_type.upper()}.png"
                concat_images(images_grey, display_img=False, output_path=os.path.join(folder_grey, filename))
                concat_images(images_color, display_img=False, output_path=os.path.join(folder_color, filename))
        else:
            images_grey = []
            images_color = []
            for block_type, attn_maps_list in cache_obj.cache.items():
                for layer_idx in range(len(attn_maps_list)):
                    attn_maps = cache_obj.aggregate_values(block_type, layer_idx)
                    images_grey, images_color = self.draw_and_append_maps(attn_maps, prompt, tokenizer, bg_image, 
                                                                            images_grey, images_color)
                
            filename = f"CA__layers.png"
            concat_images(images_grey, display_img=False, output_path=os.path.join(folder_grey, filename), 
                            num_cols=5, num_rows=12)
            concat_images(images_color, display_img=False, output_path=os.path.join(folder_color, filename), 
                            num_cols=5, num_rows=12)


    def dump_attention_maps_to_workspace(self, cache_obj, prompt, tokenizer, bg_image, output_folder, step_idx=None):
        if step_idx is not None:
            output_folder = os.path.join(output_folder, f"step_{step_idx}")
        
        if 'head' in self.dump_sources:
            self.draw_maps_per_head(cache_obj, prompt, tokenizer, bg_image, output_folder)
        if 'layer' in self.dump_sources:
            self.draw_maps_per_layer(cache_obj, prompt, tokenizer, bg_image, output_folder)
        if 'block' in self.dump_sources:
            self.draw_maps_per_block(cache_obj, prompt, tokenizer, bg_image, output_folder)
        if 'overall_mean' in self.dump_sources:
            self.draw_maps_overall_mean(cache_obj, prompt, tokenizer, bg_image, output_folder)


    def dump_attention_maps_to_workspace_per_step(self, attn_cache, prompt, tokenizer, bg_image):
        cache_types = [
            CacheTypes.CA_VISUAL_TEXT,
            # CacheTypes.CA_TEXT_VISUAL,
        ]
        for cache_type in cache_types:
            attn_maps_cache = attn_cache.get_cache(cache_type)
            output_folder = os.path.join(self.config.workspace_path, cache_type.name)
            for step_idx in attn_cache.target_timesteps:
                cache_obj = attn_maps_cache[step_idx]
                self.dump_attention_maps_to_workspace(cache_obj, prompt, tokenizer, bg_image, output_folder, step_idx)
                
            self.draw_maps_overall_mean_all_steps(attn_maps_cache, attn_cache.target_timesteps, prompt, tokenizer, 
                                                  bg_image, output_folder=output_folder)


class FluxMapsDrawerNoCache(MapsDrawerNoCache):
    def __init__(self, add_row_indices=True, res16=False, avg=False, map_size=16):
        super().__init__(add_row_indices, res16, avg, map_size)
        
    def draw_maps_overall_mean(self, objects_attn_maps, rel_objects, bg_image, output_folder):
        """ Draws the cross-attention maps averaged over every image resolution and block type """

        output_path = create_folders(self.include_color, output_folder, "overall", create_sub_folders=False)
        
        attn_maps = []
        for i in range(len(objects_attn_maps)):
            if not self.avg:
                a = objects_attn_maps[i].reshape(57, 24, self.map_size, self.map_size)
            else:
                a = objects_attn_maps[i].reshape(57, self.map_size, self.map_size).unsqueeze(dim=1)
                
            a = a.mean(dim=[0, 1])
            attn_maps.append(a)
        attn_maps = torch.stack(attn_maps, dim=0)
        
        images_grey, images_color = self.draw_and_append_maps(attn_maps, rel_objects, bg_image)
        filename = f"CA.png"
        concat_images(images_grey, display_img=False, output_path=os.path.join(output_path, f"grey__{filename}"), 
                      add_row_indices=self.add_row_indices)
        concat_images(images_color, display_img=False, output_path=os.path.join(output_path, f"color__{filename}"),
                      add_row_indices=self.add_row_indices)


    def draw_maps_per_block(self, objects_attn_maps, rel_objects, bg_image, output_folder):
        """ Draws the cross-attention maps averaged over each block (double, single) """
                
        _, folder_grey, folder_color = create_folders(self.include_color, output_folder, "per_block")
        
        # Per block, averaged over all resolutions
        block_names = ""
        images_grey = []
        images_color = []
        for block_type in FLUX_BLOCK_TYPES:
            
            attn_maps = []
            for i in range(len(objects_attn_maps)):
                if not self.avg:
                    a = objects_attn_maps[i].reshape(57, 24, self.map_size, self.map_size)
                else:
                    a = objects_attn_maps[i].reshape(57, self.map_size, self.map_size).unsqueeze(dim=1)
                start_idx = 0 if block_type == "double" else 19
                end_idx = 19 if block_type == "double" else a.shape[0]
                a = a[start_idx: end_idx].mean(dim=[0, 1])
                attn_maps.append(a)
                
            attn_maps = torch.stack(attn_maps, dim=0)
            
            images_grey, images_color = self.draw_and_append_maps(attn_maps, rel_objects, bg_image,
                                                                  images_grey, images_color)
            block_names = f"{block_names}__{block_type.upper()}"
            
        filename = f"CA__{block_names}.png"
        concat_images(images_grey, display_img=False, output_path=os.path.join(folder_grey, filename),
                      add_row_indices=self.add_row_indices)
        concat_images(images_color, display_img=False, output_path=os.path.join(folder_color, filename),
                      add_row_indices=self.add_row_indices)


    def draw_maps_per_head(self, cache_obj, prompt, tokenizer, bg_image, output_folder):
        """ Draws the heads of the cross-attention maps in each layer """
        if cache_obj is None:
            return
        
        _, folder_grey, folder_color = create_folders(self.include_color, output_folder, "per_head")
        for block_type, attn_maps_list in cache_obj.cache.items():
            for layer_idx in range(len(attn_maps_list)):
                images_grey = []
                images_color = []
                attn_maps_per_head = cache_obj.get_attention_maps(block_type, layer_idx)
                for h in range(attn_maps_per_head.shape[0]):
                    attn_maps = attn_maps_per_head[h]
                    images_grey, images_color = self.draw_and_append_maps(attn_maps, prompt, tokenizer, bg_image, 
                                                                            images_grey, images_color)

                filename = f"CA__{block_type.upper()}__L_{layer_idx}.png"
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
        for block_type in FLUX_BLOCK_TYPES:
            attn_maps = []
            for i in range(len(objects_attn_maps)):
                if not self.avg:
                    a = objects_attn_maps[i].reshape(57, 24, self.map_size, self.map_size)
                else:
                    a = objects_attn_maps[i].reshape(57, self.map_size, self.map_size).unsqueeze(dim=1)
                start_idx = 0 if block_type == "double" else 19
                end_idx = 19 if block_type == "double" else a.shape[0]
                a = a[start_idx: end_idx]
                attn_maps.append(a)
            
            attn_maps = torch.stack(attn_maps, dim=0)
            block_attn_maps_dict[block_type] = attn_maps
        
        if per_block:
            for block_type, attn_maps_tensor in block_attn_maps_dict.items():
                images_grey = []
                images_color = []
                for layer_idx in range(attn_maps_tensor.shape[1]):
                    attn_maps = attn_maps_tensor[:, layer_idx].mean(dim=1)
                    images_grey, images_color = self.draw_and_append_maps(attn_maps, rel_objects, bg_image, 
                                                                            images_grey, images_color)
                
                filename = f"CA__{block_type.upper()}.png"
                concat_images(images_grey, display_img=False, output_path=os.path.join(folder_grey, filename),
                              add_row_indices=self.add_row_indices)
                concat_images(images_color, display_img=False, output_path=os.path.join(folder_color, filename),
                              add_row_indices=self.add_row_indices)
        else:
            images_grey = []
            images_color = []
            for block_type, attn_maps_tensor in block_attn_maps_dict.items():
                for layer_idx in range(attn_maps_tensor.shape[1]):
                    attn_maps = attn_maps_tensor[:, layer_idx].mean(dim=1)
                    images_grey, images_color = self.draw_and_append_maps(attn_maps, rel_objects, bg_image, 
                                                                            images_grey, images_color)
            
            filename = f"CA__layers.png"
            concat_images(images_grey, display_img=False, output_path=os.path.join(folder_grey, filename), 
                          num_cols=5, num_rows=12, add_row_indices=self.add_row_indices)
            concat_images(images_color, display_img=False, output_path=os.path.join(folder_color, filename), 
                          num_cols=5, num_rows=12, add_row_indices=self.add_row_indices)


    def dump_attention_maps_to_workspace(self, objects_attn_maps, rel_objects, bg_image, output_folder):
        # self.draw_maps_per_head(objects_attn_maps, rel_objects, bg_image, output_folder)
        # self.draw_maps_per_layer(objects_attn_maps, rel_objects, bg_image, output_folder)
        # self.draw_maps_per_block(objects_attn_maps, rel_objects, bg_image, output_folder)
        self.draw_maps_overall_mean(objects_attn_maps, rel_objects, bg_image, output_folder)

