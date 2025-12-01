import sys
import os
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
import argparse
import shutil
from abc import ABC, abstractmethod
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from utils import (
    compute_psnr,
    extract_cross_attn_maps,
    diffusion_version_to_model_id,
)
from prompt_generator import PosNegPromptGenerator
from pipelines.flux.pipeline_flux_rf_inversion_debug import RFInversionFluxPipelineDebug
from pipelines.config import Config
from pipelines.sd.pipeline_sd2_inversion_debug import Sd2InversionPipelineDebug, load_im_into_format_from_img
from train_relation_classifier.data_creation.defs import *
from train_relation_classifier.data_creation.preprocessing.GQA_dataset.gqa_preprocessing import deserialize_structure
from train_relation_classifier.data_creation.preprocessing.GQA_dataset.gqa_dataset import load_gqa_image
from train_relation_classifier.data_creation.preprocessing.image_generation_COT_data.orm_preprocessing import load_orm_image
from train_relation_classifier.train import parse_relation_index_mapping_file


class InversionHandler(ABC):
    def __init__(self, diffusion_version, num_inference_steps, target_timesteps, map_size):
        self.model_id = diffusion_version_to_model_id(diffusion_version)
        self.num_inference_steps = num_inference_steps
        self.num_inversion_steps = num_inference_steps
        self.target_timesteps = target_timesteps
        self.map_size = map_size
        self.pipe = None  ## Initialized per-class        

    def init_config(self, output_path):
        config = Config()
        config.attention.target_timesteps = self.target_timesteps
        config.attention.dump_sources = []
        config.attention.map_size = self.map_size
        config.workspace_path = output_path
        return config
    
    @abstractmethod
    def load_inversion_pipeline(self):
        pass
    
    @abstractmethod
    def invert_image(self, input_img, src_prompt, output_path, dump_images=False):
        pass
    
    @abstractmethod
    def get_and_dump_cross_attn_maps_paths(self, src_prompt, obj1, obj2, rel_name, 
                                            timestep, flux_output_path, flatten_maps=True,
                                            output_map_resolution = -1, target_map_size=16):
        pass
    

class FluxHandler(InversionHandler):
    def __init__(self, diffusion_version, target_timesteps, map_size):
        if "schnell" in diffusion_version:
            num_inference_steps = 4
            if len(target_timesteps) == 0:
                target_timesteps = [1, 3]
        else:
            num_inference_steps = 50
            if len(target_timesteps) == 0:
                target_timesteps = [5, 10, 15, 20, 25, 49]
        
        super().__init__(diffusion_version, num_inference_steps, target_timesteps, map_size)
        self.pipe = self.load_inversion_pipeline()

    
    def load_inversion_pipeline(self):        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pipe = RFInversionFluxPipelineDebug.from_pretrained(self.model_id, torch_dtype=torch.bfloat16)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        return pipe
    
    
    def get_initial_latents(self, input_img, src_prompt, height, width, generator):
        inverted_latents, image_latents, latent_image_ids = self.pipe.invert(
            input_img,
            src_prompt,
            num_inversion_steps=self.num_inversion_steps,
            gamma=1.0,
            height=height,
            width=width,
        )
        return inverted_latents, image_latents, latent_image_ids
            
    
    def invert_image(self, input_img, src_prompt, output_path, dump_images=False):
        visual_tokens = self.map_size * 16
        input_img = input_img.resize((visual_tokens, visual_tokens))
        
        seed = np.random.randint(0, 1000000)
        g_cpu = torch.Generator().manual_seed(seed)
        inverted_latents, image_latents, latent_image_ids = self.get_initial_latents(input_img, src_prompt, 
                                                                                     visual_tokens, visual_tokens,
                                                                                     generator=g_cpu) 

        config = self.init_config(output_path)
        text_tokens = 512
        inverted_img = self.pipe(
            src_prompt,
            inverted_latents=inverted_latents,
            image_latents=image_latents,
            latent_image_ids=latent_image_ids,
            start_timestep=0,
            stop_timestep=1,
            height=visual_tokens,
            width=visual_tokens,
            guidance_scale=0,
            num_inference_steps=self.num_inference_steps,
            max_sequence_length=text_tokens,
            generator=g_cpu,
            input_config=config
        ).images[0]

        psnr = compute_psnr(input_img, inverted_img)
        inversion_metadata = {
            "num_inversion_steps": self.num_inversion_steps,
            "psnr": psnr,
        }
        inversion_metadata_file = os.path.join(config.workspace_path, "inversion_metadata.json")
        with open(inversion_metadata_file, "w") as f:
            json.dump(inversion_metadata, f, indent=4)
        
        return inverted_img, psnr 
    
    
    def get_and_dump_cross_attn_maps_paths(self, src_prompt, obj1, obj2, rel_name, # type: ignore
                                            timestep, flux_output_path, flatten_maps=True, 
                                            output_map_resolution = -1, target_map_size=16):
        rel_items_cross_attn_maps = extract_cross_attn_maps(self.pipe.tokenizer_2, self.pipe.attention_cache, 
                                                            src_prompt, obj1, obj2, rel_name, timestep, 
                                                            flatten_maps=flatten_maps, 
                                                            output_map_resolution=output_map_resolution,
                                                            target_map_size=target_map_size)
        cross_attn_maps_paths = []
        for i, attn_maps in enumerate(rel_items_cross_attn_maps):
            filename = f"cross_attn__T_{timestep}__obj_{i}"
            attn_map_path = os.path.join(flux_output_path, f"{filename}.pth")
            cross_attn_maps_paths.append(attn_map_path)
            torch.save(attn_maps, attn_map_path)
            
        return cross_attn_maps_paths
    

class Sd2Handler(InversionHandler):
    def __init__(self, diffusion_version, target_timesteps, map_size):
        num_inference_steps = 50
        if len(target_timesteps) == 0:
            target_timesteps = [5, 10, 15, 20, 25, 49]
        super().__init__(diffusion_version, num_inference_steps, target_timesteps, map_size)
        self.pipe = self.load_inversion_pipeline()
        self.guidance_scale = 1.0


    def load_inversion_pipeline(self): # type: ignore
        from diffusers import DDIMScheduler
        
        scheduler = DDIMScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        pipe = Sd2InversionPipelineDebug.from_pretrained(
            self.model_id,
            scheduler=scheduler,
            safety_checker=None,
        ).to('cuda')
        pipe.set_progress_bar_config(disable=True)
        return pipe
    
    def get_initial_latents(self, input_img, src_prompt, generator):
        latents = self.pipe.invert(
            src_prompt, 
            latents=None, 
            image=input_img,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale, 
            num_iter=5,
        ).latents
        return latents
    
    def invert_image(self, input_img, src_prompt, output_path, dump_images=False): # type: ignore
        input_img = load_im_into_format_from_img(input_img)
        width, height = input_img.size
        
        config = self.init_config(output_path)

        seed = np.random.randint(0, 1000000)
        g_cpu = torch.Generator().manual_seed(seed)
        
        with torch.no_grad():
            # RUN FP inversion on vae_latent
            latents = self.get_initial_latents(input_img, src_prompt, generator=g_cpu)
            inverted_img = self.pipe(
                prompt=[src_prompt], 
                latents=latents, 
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps, 
                output_type='pil',
                width=width,
                height=height,
                start_timestep=0,
                input_config=config,
            ).images[0]
        
        psnr = compute_psnr(input_img, inverted_img)
        inversion_metadata = {
            # "workspace_path": config.workspace_path,
            # "prompt": src_prompt,
            "seed": seed,
            "num_inversion_steps": self.num_inversion_steps,
            "psnr": psnr,
        }
        inversion_metadata_file = os.path.join(config.workspace_path, "inversion_metadata.json")
        with open(inversion_metadata_file, "w") as f:
            json.dump(inversion_metadata, f, indent=4)
        
        return inverted_img, psnr
            
    
    def get_and_dump_cross_attn_maps_paths(self, src_prompt, obj1, obj2, rel_name, # type: ignore
                                            timestep, flux_output_path, flatten_maps=True,
                                            output_map_resolution = -1, target_map_size=16):
        rel_items_cross_attn_maps = extract_cross_attn_maps(self.pipe.tokenizer, self.pipe.attention_cache,
                                                            src_prompt, obj1, obj2, rel_name, 
                                                            timestep, flatten_maps=flatten_maps,
                                                            remove_start_token=True, 
                                                            output_map_resolution=output_map_resolution,
                                                            target_map_size=target_map_size)
        cross_attn_maps_paths = []
        for i, attn_maps in enumerate(rel_items_cross_attn_maps):
            filename = f"cross_attn__T_{timestep}__obj_{i}"
            attn_map_path = os.path.join(flux_output_path, f"{filename}.pth")
            cross_attn_maps_paths.append(attn_map_path)
            torch.save(attn_maps, attn_map_path)

        return cross_attn_maps_paths
    
    
def init_inversion_handler(diffusion_version, target_timesteps, map_size):
    if "schnell" in diffusion_version or "dev" in diffusion_version:
        handler = FluxHandler(diffusion_version, target_timesteps, map_size)
    elif "sd2" in diffusion_version or "sd14" in diffusion_version:
        handler = Sd2Handler(diffusion_version, target_timesteps, map_size)
    else:
        print("Using default diffusion model: Flux-schnell")
        handler = FluxHandler("schnell", target_timesteps, map_size)
        
    return handler


def get_load_img_function(data_src, gqa_images_root, orm_images_root):
    """ Given a dataset source, returns the appropriate `load_img` function. """
    if data_src == DATA_SOURCE_GQA:
        load_img_func = lambda img_id: load_gqa_image(img_id, root_dir=gqa_images_root)
    elif data_src == DATA_SOURCE_ORM:
        load_img_func = lambda img_id: load_orm_image(img_id, root_dir=orm_images_root)
    else:
        load_img_func = None
        print(f"Invalid data source: {data_src}")

    return load_img_func


def run_on_dataset_file(diffusion_version, map_size, dataset_path, mapping_path, data_src, 
                        output_map_resolution, target_timesteps, 
                        target_map_size, output_path, gqa_images_root, orm_images_root,
                        min_psnr=20, dump_images=False, flatten_maps=True):
    """
    For each image and for each relation triplet:
       1. Generate a source prompt from the triplet
       2. Invert the image with the source prompt
       3. Extract the relevant cross-attention maps
       4. Save metadata of the sample.
    """
    
    inversion_handler = init_inversion_handler(diffusion_version, target_timesteps=target_timesteps, 
                                               map_size=map_size)
    dataset = deserialize_structure(dataset_path)
    rel_name_to_idx, _ = parse_relation_index_mapping_file(mapping_path)
    load_img_function = get_load_img_function(data_src, gqa_images_root, orm_images_root)
    
    all_relation_names = set(rel_name_to_idx.keys())
    prompt_generator = PosNegPromptGenerator(all_relation_names, num_prompts=1)
        
    if diffusion_version == "sd2" or diffusion_version == "sd14":
        min_psnr = 19.5
    
    num_below_psnr_threshold = 0
    samples_metadata = []
    for k, item in enumerate(tqdm(dataset)):
        img_id, orig_obj1, orig_obj2, orig_rel_name = item[:4]
        
        # Load the input image
        input_img = load_img_function(img_id)
            
        src_prompts = prompt_generator.generate_source_prompts(orig_obj1, orig_obj2, orig_rel_name)
        for src_prompt, prompt_type, gt_rel_name, rel_objects in src_prompts:
            obj1, obj2 = rel_objects
            inverted_img, psnr = inversion_handler.invert_image(input_img, src_prompt, output_path, 
                                                                dump_images=dump_images)
            if psnr < min_psnr:
                print(f"PSNR value is too low: {psnr} for {img_id} {src_prompt}. Skipping.")
                num_below_psnr_threshold += 1
                break  
            
            flux_output_path = inversion_handler.pipe.conf.workspace_path
            for timestep in inversion_handler.target_timesteps:
                cross_attn_maps_paths = inversion_handler.get_and_dump_cross_attn_maps_paths(src_prompt, obj1, obj2, 
                                                                                            gt_rel_name, timestep,
                                                                                            flux_output_path,
                                                                                            flatten_maps=flatten_maps,
                                                                                            output_map_resolution=output_map_resolution,
                                                                                            target_map_size=target_map_size)
                sample_metadata = {
                    "flux_output_path": flux_output_path,
                    "prompt": src_prompt,
                    "timestep": timestep,
                    "cross_attn_maps_paths": cross_attn_maps_paths,
                    "rel_objects": [obj1, obj2],
                    "gt_relation_name": gt_rel_name,
                    "img_id": img_id,
                    "prompt_type": prompt_type,
                }
                samples_metadata.append(sample_metadata)
    print(f"Total below PSNR threshold: {num_below_psnr_threshold}")
    
    if len(samples_metadata) == 0:
        print("No samples")
        return None
    
    samples_metadata_path = os.path.join(output_path, RESULT_DATA_FILENAME)
    with open(samples_metadata_path, "w") as f:
        json.dump(samples_metadata, f, indent=4)
        
    return samples_metadata_path
        

def parse_metadata_file(metadata_path):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    dataset_path = metadata[FLAT_DATA_PATH_KEY]
    mapping_path = metadata[MAPPING_PATH_KEY]
    data_src = metadata[DATA_SOURCE_KEY] if DATA_SOURCE_KEY in metadata else DATA_SOURCE_GQA
    return dataset_path, mapping_path, data_src


def dump_metadata(data_path, mapping_path, output_path, map_size, target_map_size):
    # Copy mapping file to output directory
    mapping_filename = os.path.basename(mapping_path)
    local_mapping_path = os.path.join(output_path, mapping_filename)
    shutil.copy(mapping_path, local_mapping_path)
    
    result_metadata = {
        RESULT_DATA_PATH_KEY: data_path,
        MAPPING_PATH_KEY: local_mapping_path,  # Now points to local copy
        MAP_SIZE_KEY: map_size,
        TARGET_MAP_SIZE_KEY: target_map_size,
    }
    with open(os.path.join(output_path, RESULT_METADATA_FILENAME), "w") as f:
        json.dump(result_metadata, f, indent=4)
    

def run_on_metadata_file(diffusion_version, map_size, metadata_path, 
                         output_map_resolution, target_timesteps, 
                         target_map_size, output_path, gqa_images_root, orm_images_root,
                         min_psnr=20, dump_images=False):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = f"tmp_run__{timestamp}"
    output_path = os.path.join(output_path, fn)
    os.makedirs(output_path, exist_ok=True)
    
    dataset_path, mapping_path, data_src = parse_metadata_file(metadata_path)
        
    result_data_path = run_on_dataset_file(diffusion_version, map_size, dataset_path, mapping_path, data_src,
                                           output_map_resolution, target_timesteps, 
                                           target_map_size, output_path, gqa_images_root,
                                           orm_images_root, min_psnr, dump_images) 
    dump_metadata(result_data_path, mapping_path, output_path, map_size, target_map_size)


def prepare_chunks(entire_dataset, num_gpus, output_path):
    """Divide dataset into chunks for parallel processing."""
    chunks_json_path = os.path.join(output_path, "chunks")
    os.makedirs(chunks_json_path, exist_ok=True)
    
    chunks_paths = []
    base_chunk_size = len(entire_dataset) // num_gpus
    chunk_remainder = len(entire_dataset) % num_gpus
    chunk_size_per_gpu = [base_chunk_size for _ in range(num_gpus)]
    if chunk_remainder != 0:
        chunk_size_per_gpu[-1] = base_chunk_size + chunk_remainder
    
    for i, chunk_size in enumerate(chunk_size_per_gpu):
        start_idx = sum(chunk_size_per_gpu[:i])
        end_idx = start_idx + chunk_size
        chunk_dataset = entire_dataset[start_idx:end_idx]
        chunk_filename = f"chunk_{i}__size_{chunk_size}.json"
        chunk_path = os.path.join(chunks_json_path, chunk_filename)
        chunks_paths.append(chunk_path)
        with open(chunk_path, "w") as f:
            json.dump(chunk_dataset, f, indent=4)
        
    return chunks_paths


def run_chunk_parallel(args, dataset_path, mapping_path, data_src, 
                       output_path, gpu_idx):
    """Run a single chunk on a specific GPU (for multiprocessing worker)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    run_on_dataset_file(
        args.diffusion_version, args.map_size, dataset_path, mapping_path, data_src, 
        args.output_map_resolution, 
        args.target_timesteps, 
        args.target_map_size, output_path,
        args.gqa_images_root,
        args.orm_images_root,
    )


def worker(task_args):
    """Wrapper for multiprocessing worker to unpack task arguments."""
    try:
        run_chunk_parallel(*task_args)
    except Exception as e:
        print(f"Error in worker: {e}")


def run_parallel(args, metadata_path, output_path, gpu_indices, experiment_name=""):
    """
    1. Divide the dataset into chunks according to the number of GPUs available
    2. Generate each chunk in parallel
    3. Merge results
    """
    
    # Create dedicated folder
    if args.use_date_str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = f"run_parallel__{timestamp}"
        fn += f"__{experiment_name}" if len(experiment_name) > 0 else ""
    else:
        fn = f"{experiment_name}"
        
    output_path = os.path.join(output_path, fn)
    os.makedirs(output_path, exist_ok=True)
    
    # Dump the input args
    if args is not None:
        with open(os.path.join(output_path, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
    
    # Divide the dataset into chunks and run each chunk in a different process
    dataset_path, mapping_path, data_src = parse_metadata_file(metadata_path)
    entire_dataset = deserialize_structure(dataset_path)
    num_gpus = len(gpu_indices)
    chunks_paths = prepare_chunks(entire_dataset, num_gpus, output_path)
    
    task_args = []
    all_chunk_results_paths = []
    for i, (chunk_dataset_path, gpu_idx) in enumerate(zip(chunks_paths, gpu_indices)):
        chunk_output_path = os.path.join(output_path, f"chunk_{i}__GPU_{gpu_idx}")
        t_args = (
            args, chunk_dataset_path, mapping_path, data_src, 
            chunk_output_path, gpu_idx
        )
        task_args.append(t_args)
        chunk_result_path = os.path.join(chunk_output_path, RESULT_DATA_FILENAME)
        all_chunk_results_paths.append(chunk_result_path)
    
    # Use torch.multiprocessing to spawn processes
    mp.set_start_method("spawn", force=True)
    with mp.Pool(processes=num_gpus) as pool:
        try:
            pool.map(worker, task_args)
        except Exception as e:
            print(f"Error during processing: {e}")
        finally:
            pool.close()
            pool.join()
    
    # Merge the chunks' results, dump the final data json, and dump the results metadata
    final_output = []
    for chunk_result_path in all_chunk_results_paths:
        with open(chunk_result_path, "r") as f:
            data = json.loads(f.read())
        final_output += data
    
    final_data_output_path = os.path.join(output_path, RESULT_DATA_FILENAME)
    with open(final_data_output_path, "w") as f:
        json.dump(final_output, f, indent=4)
    
    dump_metadata(final_data_output_path, mapping_path, output_path, args.map_size, args.target_map_size)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_path', default="", 
                        help="Path to a GQA preprocessed dataset metadata file, which includes both the data file and the mapping file")
    parser.add_argument('--output_path', default="", help="Output path of where the pretraining data will be kept")
    parser.add_argument('--diffusion_version', default="schnell", help="Either schnell or dev")
    parser.add_argument('--map_size', default=16, type=int, help="Cross-attention HxW map size. Default=16x16")
    parser.add_argument('--gpu_indices', default=[], type=int, nargs="+", help="List of available GPU indices")
    parser.add_argument('--target_timesteps', default=[], type=int, nargs="+", help="List of available GPU indices")
    parser.add_argument('--experiment_name', default="", help="The name of the current experiment")
    parser.add_argument('--use_date_str', action='store_true',
                        help="whether to use the current date and append it to the experiment name")
    parser.add_argument('--output_map_resolution', default = -1, type=int, 
                        help="Output cross-attn maps only with this resolution. Default = -1, meaning use all resolutions")
    parser.add_argument('--target_map_size', default = 16, type=int, 
                        help="Downsample all attention maps to this target size. Default = 16.")
    parser.add_argument('--gqa_images_root', default=GQA_IMAGES_PATH, type=str,
                        help="Root path to GQA images. If not provided, uses default from defs.")
    parser.add_argument('--orm_images_root', default=ORM_IMAGES_PATH, type=str,
                        help="Root path to ORM images. If not provided, uses default from defs.")
    
    args = parser.parse_args()
    args.diffusion_version = args.diffusion_version.lower()
    args.map_size = int(args.map_size)
    args.output_map_resolution = int(args.output_map_resolution)
    args.target_map_size = int(args.target_map_size)
    args.target_timesteps = [int(timestep) for timestep in args.target_timesteps]
    print(f"{args=}")
    return args


if __name__ == "__main__":
    args = parse_args()
    
    # Use parallel processing if multiple GPUs specified
    if len(args.gpu_indices) > 1:
        run_parallel(args, args.metadata_path, args.output_path, args.gpu_indices, args.experiment_name)
    else:
        # Sequential processing
        gpu_idx = args.gpu_indices[0] if len(args.gpu_indices) == 1 else 0
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_idx}"
        run_on_metadata_file(
            args.diffusion_version, args.map_size, args.metadata_path, 
            args.output_map_resolution, args.target_timesteps, 
            args.target_map_size, args.output_path,
            args.gqa_images_root,
            args.orm_images_root,
        )
        