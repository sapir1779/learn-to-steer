import os
import json
import numpy as np
from datetime import datetime
import uuid
import torch
import torch.nn.functional as F


def compute_psnr(orig, inverted, max_pixel_value=1):
    def norm_img(img):
        img = np.array(img) / 255.0
        img = img.astype(np.float32)
        return img
    
    mse = np.mean((norm_img(orig) - norm_img(inverted)) ** 2)
    if mse == 0:
        return float('inf')  # Perfect match
    
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr


def process_pipe_config(pipe, input_config):
    pipe.conf = input_config
    if not pipe.conf or not pipe.conf.debug:
        return
    
    if pipe.conf.use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{uuid.uuid4().hex[:6]}"
        pipe.conf.workspace_path = os.path.join(pipe.conf.workspace_path, timestamp)
    os.makedirs(pipe.conf.workspace_path, exist_ok=True)
    
    pipe.conf.dump(pipe.conf.workspace_path)
    
    metadata_file = os.path.join(pipe.conf.workspace_path, "diffusion_metadata.json")
    metadata = {
        "workspace_path": pipe.conf.workspace_path,
        "prompt": pipe.prompt,
        "seed": pipe.seed,
        "num_inference_steps": pipe.num_inference_steps,
        "model": pipe.config._name_or_path,
        "img_width": pipe.img_width,
        "img_height": pipe.img_height,
        "max_sequence_length": pipe.max_sequence_length,
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)


def resize_and_concat_attn_maps_from_dict(attn_maps_dict, target_size=16, renormalize=True):
    """
    attn_maps_dict: dict mapping resolution (int) -> tensor of shape (L, H, H_in, W_in)
    renormalize: if True, normalize each map so that sum over (H, W) = 1 after resizing
    Returns:
        - concatenated_maps: (total_LxH, target_size, target_size)
        - source_res_list: list of resolution per map
    """
    
    all_maps = []
    for res in sorted(attn_maps_dict.keys()):
        attn = attn_maps_dict[res]  # (L, H, H_in, W_in)
        LH, H_in, W_in = attn.shape

        attn = attn.view(LH, 1, H_in, W_in)
        resized = F.interpolate(attn, size=(target_size, target_size), mode='bilinear', align_corners=False)

        if renormalize:
            resized = resized / (resized.sum(dim=(-2, -1), keepdim=True) + 1e-8)

        all_maps.append(resized.squeeze(1))  # (L*H, target_size, target_size)

    concatenated_maps = torch.cat(all_maps, dim=0)  # (total_L*H, target_size, target_size)
    return concatenated_maps


def find_word_token_inds(tokenizer, prompt, word, remove_start_token=False):
    tokens = np.array(tokenizer.encode(prompt))
    all_word_tokens = tokenizer.encode(word)[:-1]  # Get the tokens for the word (excluding </s>)
    if remove_start_token and len(all_word_tokens) > 1:
        all_word_tokens = all_word_tokens[1:]
    
    has_several_tokens = len(all_word_tokens) > 1
    if has_several_tokens:
        start_idx = 0
        for i, token in enumerate(all_word_tokens):
            if word[0] in tokenizer.decode(token):
                start_idx = i
                break
        all_word_tokens = all_word_tokens[start_idx:]

    # Find the index of the matching token
    output = []
    token_idx = np.where(tokens == all_word_tokens[0])[0]
    if token_idx.size > 0:
        for j in range(len(token_idx)):
            word_token_inds = [token_idx[j] + i for i in range(len(all_word_tokens))]
            output.append(word_token_inds)

    return output


def get_word_cross_attn_maps(tokenizer, attention_cache, prompt, word, timestep, 
                             layer_start=-1, layer_end=-1, instance_idx=0, add_batch_dim=False,
                             output_all_heads_and_layers=True,
                             flatten_maps=True, remove_start_token=False,
                             output_map_resolution = -1, target_map_size=16,
    ) -> torch.Tensor:
    ## Get the token indices of the instances of the given object
    instances_token_inds = find_word_token_inds(tokenizer, prompt, word, remove_start_token)
    if len(instances_token_inds) == 0 or instance_idx >= len(instances_token_inds):
        return None
    
    ## Get the cross-attention maps of the word
    word_token_inds = instances_token_inds[instance_idx]
    vis_text_attn_cache = attention_cache.get_cross_attn_visual_text_cache()[timestep]
    # (57, 24, N, 16, 16), where N = len(word_token_inds)
    all_layers_word_cross_attn_maps = vis_text_attn_cache.get_attn_maps_from_indices(word_token_inds)
    
    output = {}
    all_resolutions = list(all_layers_word_cross_attn_maps.keys())
    for res_key in all_resolutions:
        if output_map_resolution > 0 and res_key != output_map_resolution:
            continue
        
        word_cross_attn_maps = all_layers_word_cross_attn_maps[res_key]
        if len(word_token_inds) > 1:
            # (57, 24, N, 16, 16) --> (57, 24, 16, 16)
            word_cross_attn_maps = torch.mean(word_cross_attn_maps, dim=2)
        else:
            # (57, 24, 1, 16, 16) --> (57, 24, 16, 16)
            word_cross_attn_maps = word_cross_attn_maps.squeeze(dim=2)
        
        if not output_all_heads_and_layers and layer_start >= 0 and layer_end > layer_start:
            ## Extract the specific layers that describe the word
            # (57, 24, 16, 16) --> (L, 24, 16, 16), where L = (layer_end - layer_start)
            word_cross_attn_maps = word_cross_attn_maps[layer_start: layer_end]
            _, _, height, width = word_cross_attn_maps.shape
            # (L, 24, 16, 16) --> (L * 24, 16, 16)
            word_cross_attn_maps = word_cross_attn_maps.view(-1, height, width)
            # (L * 24, 16, 16) --> (16, 16)
            word_cross_attn_maps = torch.mean(word_cross_attn_maps, dim=0)
        
        if add_batch_dim:
            word_cross_attn_maps = word_cross_attn_maps.unsqueeze(0)
        elif flatten_maps:
                layers, heads, height, width = word_cross_attn_maps.shape
                word_cross_attn_maps = word_cross_attn_maps.reshape(layers * heads, height, width)
        
        output[res_key] = word_cross_attn_maps
    
    should_resize = len(all_resolutions) > 1 or all_resolutions[0] != target_map_size
    if should_resize:
        final_word_cross_attn_maps = resize_and_concat_attn_maps_from_dict(output, target_map_size)
    else:
        final_word_cross_attn_maps = output[all_resolutions[0]]
        
    return final_word_cross_attn_maps


def extract_cross_attn_maps(tokenizer, attn_cache, prompt, obj1, obj2, rel_name, timestep, 
                            add_batch_dim=False, output_all_heads_and_layers=True, flatten_maps=True,
                            remove_start_token=False, output_map_resolution = -1,
                            target_map_size=16):
    cross_attn1 = get_word_cross_attn_maps(tokenizer, attn_cache, prompt, obj1, timestep, 
                                           add_batch_dim=add_batch_dim, 
                                           output_all_heads_and_layers=output_all_heads_and_layers,
                                           flatten_maps=flatten_maps,
                                           remove_start_token=remove_start_token,
                                           output_map_resolution=output_map_resolution,
                                           target_map_size=target_map_size)
    cross_attn2 = get_word_cross_attn_maps(tokenizer, attn_cache, prompt, obj2, timestep, 
                                           add_batch_dim=add_batch_dim, 
                                           output_all_heads_and_layers=output_all_heads_and_layers,
                                           flatten_maps=flatten_maps,
                                           remove_start_token=remove_start_token,
                                           output_map_resolution=output_map_resolution,
                                           target_map_size=target_map_size)
    output_maps = [cross_attn1, cross_attn2]
    return output_maps


FLUX_SCHNELL_DIFFUSION_VERSION = "schnell"
FLUX_DEV_DIFFUSION_VERSION = "dev"
SD2_DIFFUSION_VERSION = "sd2"
SD14_DIFFUSION_VERSION = "sd14"


def diffusion_version_to_model_id(version):
    if "schnell" in version:
        model_id = "black-forest-labs/FLUX.1-schnell"
    elif "dev" in version:
        model_id = "black-forest-labs/FLUX.1-dev"
    elif "sd2" in version:
        model_id = "stabilityai/stable-diffusion-2-1-base"
    elif "sd14" in version:
        model_id = "CompVis/stable-diffusion-v1-4"
    elif "rrnet" in version:
        model_id = ""
        print("Using RRNet")
    elif "initno" in version:
        model_id = ""
        print("Using InitNO")
    elif "for" in version:
        model_id = ""
        print("Using FOR")
    else:
        model_id = "black-forest-labs/FLUX.1-schnell"
        print("Using default diffusion model: Flux-schnell")

    print(f"Using diffusion model: {version} --> {model_id}")
    return model_id
