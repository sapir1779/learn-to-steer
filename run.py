import os
import sys
import argparse
from lightning import seed_everything
import torch
from abc import ABC, abstractmethod

from pipelines.config import Config
from train_relation_classifier.model_utils import load_model
from utils import (
    diffusion_version_to_model_id,
    FLUX_SCHNELL_DIFFUSION_VERSION,
    FLUX_DEV_DIFFUSION_VERSION,
    SD2_DIFFUSION_VERSION,
    SD14_DIFFUSION_VERSION,   
)
from prompt_parsing_utils import PromptParser


# ==================== Helper Functions ====================

def load_relation_classifier(relation_classifier_model_path, device="cuda"):
    """Load the relation classifier from checkpoint."""
    relation_classifier = load_model(relation_classifier_model_path, device)
    relation_classifier = relation_classifier.to(device)
    return relation_classifier


def get_orig_map_size(relation_classifier):
    """Extract the original map size from the relation classifier's training config."""
    orig_map_size = -1
    if relation_classifier is not None and relation_classifier.training_config is not None \
        and "orig_map_size" in relation_classifier.training_config:
        orig_map_size = relation_classifier.training_config["orig_map_size"]
    
    return orig_map_size


def get_test_time_opt_config(diffusion_version):
    if diffusion_version == FLUX_SCHNELL_DIFFUSION_VERSION:
        output_config = {
            "step_sizes": [5],
            "max_refinement_steps": [15],
            "end_idx": 1,
        }
    elif diffusion_version == FLUX_DEV_DIFFUSION_VERSION:
        output_config = {
            "step_sizes": [5] * 25,
            "max_refinement_steps": [15] * 25,
            "end_idx": 25,
        }
    elif diffusion_version == SD2_DIFFUSION_VERSION:
        output_config = {
            "step_sizes": [5] * 25,
            "max_refinement_steps": [15] * 25,
            "end_idx": 25,
        }
    elif diffusion_version == SD14_DIFFUSION_VERSION:
        output_config = {
            "step_sizes": [7.5] * 25,
            "max_refinement_steps": [15] * 25,
            "end_idx": 25,
        }
    else:
        output_config = {
            "step_sizes": [5],
            "max_refinement_steps": [15],
            "end_idx": 1,
        }
    
    return output_config


# ==================== ImageGenerator Classes ====================

class ImageGenerator(ABC):
    """Base class for image generation with relation classifier guidance."""
    
    def __init__(self, args, num_inference_steps, guidance_scale, map_size=None):
        self.args = args
        self.diffusion_version = args.diffusion_version
        self.model_id = diffusion_version_to_model_id(self.diffusion_version)
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        
        # Load relation classifier
        self.relation_classifier = load_relation_classifier(args.relation_classifier_model_path)
        
        orig_map_size = get_orig_map_size(self.relation_classifier)
        if orig_map_size > 0:
            self.map_size = orig_map_size
        elif map_size is not None:
            self.map_size = map_size
        else:
            self.map_size = args.map_size
        
        self.test_time_opt_config = get_test_time_opt_config(args.diffusion_version)
        
        self.pipe = None  # Initialized per-class
        self.maps_drawer_class = None  # Initialized per-class
        self.prompt_parser = PromptParser()


    @abstractmethod
    def load_pipeline(self):
        """Load the diffusion model pipeline. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def generate_image(self, prompt, seed, workspace_path, sample_prefix, save_image=False):
        """Generate an image from a prompt."""
        pass
    
    def init_config(self, workspace_path, sample_prefix, use_timestep=False):
        """Initialize the Config object for generation."""
        config = Config()
        
        config.num_inference_steps = self.num_inference_steps
        # Only visualize cross-attention maps at the final timestep
        config.attention.target_timesteps = [self.num_inference_steps - 1] if self.args.visualize_cross_attn_maps else []
        config.attention.dump_sources = ["overall_mean", "resolution"] if self.args.visualize_cross_attn_maps else []
        config.attention.map_size = self.map_size
        
        # Test-time optimization
        config.test_time_optimization.enable = True
        config.test_time_optimization.step_sizes = self.test_time_opt_config["step_sizes"]
        config.test_time_optimization.iterative_refinement.max_refinement_steps = self.test_time_opt_config["max_refinement_steps"]
        config.test_time_optimization.end_idx = self.test_time_opt_config["end_idx"]
        config.test_time_optimization.test_classification = False
        
        # General
        config.workspace_path = workspace_path
        config.sample_prefix = sample_prefix
        config.debug = True
        config.use_timestamp = use_timestep
        
        return config
    
    
    def dump_attention_maps(self, attn_cache, prompt, tokenizer, bg_image):
        """Dump attention maps using the appropriate drawer class."""
        obj1, _, obj2 = self.prompt_parser.extract_relation_triplets_from_prompt(prompt)[0]
        objects = [obj1, obj2]
        drawer = self.maps_drawer_class(self.pipe.conf, skip_prefix_tokens=True, objects=objects)
        drawer.dump_attention_maps_to_workspace_per_step(attn_cache, prompt, tokenizer, bg_image)


class FluxImageGenerator(ImageGenerator):
    def __init__(self, args):
        if FLUX_SCHNELL_DIFFUSION_VERSION in args.diffusion_version:
            num_inference_steps = 4
            guidance_scale = 0.0
            map_size = None
        else:
            num_inference_steps = 50
            guidance_scale = 3.5
            map_size = 32
            
        super().__init__(args, num_inference_steps, guidance_scale, map_size)
        self.pipe = self.load_pipeline()
        
        from pipelines.flux.flux_drawing_utils import FluxMapsDrawer
        self.maps_drawer_class = FluxMapsDrawer
        
        
    def load_pipeline(self):
        from pipelines.flux.pipeline_flux_debug import FluxPipelineDebug
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pipe = FluxPipelineDebug.from_pretrained(self.model_id, torch_dtype=torch.bfloat16)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        return pipe
    
    
    def generate_image(self, prompt, seed, workspace_path, sample_prefix, save_image=False):
        config = self.init_config(workspace_path, sample_prefix, save_image)
        g_cpu = torch.Generator().manual_seed(seed) if seed >= 0 else None
        text_tokens = 512
        visual_tokens = self.map_size * 16
        
        image = self.pipe(
            prompt,
            height=visual_tokens,
            width=visual_tokens,
            guidance_scale=self.guidance_scale,
            num_inference_steps=config.num_inference_steps,
            max_sequence_length=text_tokens,
            generator=g_cpu,
            input_config=config,
            relation_classifier=self.relation_classifier,
            prompt_parser=self.prompt_parser,
        ).images[0]
        
        if self.args.visualize_cross_attn_maps:
            self.dump_attention_maps(self.pipe.attention_cache, self.pipe.prompt, 
                                    self.pipe.tokenizer_2, bg_image=image)
        if save_image:
            image.save(os.path.join(self.pipe.conf.workspace_path, "generated_image.png"))
        
        return image
    

class Sd2ImageGenerator(ImageGenerator):
    def __init__(self, args):
        num_inference_steps = 50
        guidance_scale = 7.5
        super().__init__(args, num_inference_steps, guidance_scale)
        
        self.pipe = self.load_pipeline()
        
        from pipelines.sd.sd2_drawing_utils import Sd2MapsDrawer
        self.maps_drawer_class = Sd2MapsDrawer
        
        
    def load_pipeline(self):
        from diffusers import DDIMScheduler
        from pipelines.sd.pipeline_sd2_debug import Sd2PipelineDebug
        
        scheduler = DDIMScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        pipe = Sd2PipelineDebug.from_pretrained(
            self.model_id,
            scheduler=scheduler,
            safety_checker=None,
        ).to('cuda')
        pipe.set_progress_bar_config(disable=True)
        return pipe
    
    
    def generate_image(self, prompt, seed, workspace_path, sample_prefix, save_image=False):
        config = self.init_config(workspace_path, sample_prefix, save_image)
        g_cpu = torch.Generator().manual_seed(seed) if seed >= 0 else None
        
        height = width = 512
        image = self.pipe(
            prompt,
            height=height,
            width=width,
            num_inference_steps=config.num_inference_steps,
            generator=g_cpu, 
            guidance_scale=self.guidance_scale,
            input_config=config,
            relation_classifier=self.relation_classifier,
            prompt_parser=self.prompt_parser,
        ).images[0]
        
        if self.args.visualize_cross_attn_maps:
            self.dump_attention_maps(self.pipe.attention_cache, self.pipe.prompt, 
                                     self.pipe.tokenizer, bg_image=image)
        
        if save_image:
            image.save(os.path.join(self.pipe.conf.workspace_path, "generated_image.png"))
        
        return image


def init_image_generator(args) -> ImageGenerator:    
    if FLUX_SCHNELL_DIFFUSION_VERSION in args.diffusion_version or \
        FLUX_DEV_DIFFUSION_VERSION in args.diffusion_version:
        img_generator = FluxImageGenerator(args)
        
    elif SD2_DIFFUSION_VERSION in args.diffusion_version or \
        SD14_DIFFUSION_VERSION in args.diffusion_version:
        img_generator = Sd2ImageGenerator(args)
        
    else:
        print("Using the default ImageGenerator: flux schnell")
        img_generator = FluxImageGenerator(args)
    
    return img_generator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images with spatial relation guidance using trained relation classifiers",
    )
    
    parser.add_argument(
        "--diffusion_version",
        type=str,
        choices=["schnell", "dev", "sd2", "sd14"],
        required=True,
        help="Diffusion version to use for generation"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt describing the desired image"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save generated images (uses default if not provided)"
    )
    parser.add_argument(
        "--relation_classifier_model_path",
        type=str,
        default=None,
        help="Path to trained relation classifier checkpoint (uses default if not provided)"
    )
    parser.add_argument(
        "--map_size",
        type=int,
        default=16,
        help="Map size for cross-attn maps (and image resolution implied) (default: 16)"
    )
    parser.add_argument(
        "--gpu_idx",
        type=int,
        default=0,
        help="GPU index to use (default: 0)"
    )
    parser.add_argument(
        "--visualize_cross_attn_maps",
        action="store_true",
        help="Whether to visualize cross-attention maps during generation"
    )
    
    args = parser.parse_args()
    args.diffusion_version = args.diffusion_version.lower()
    
    # Use provided checkpoint or default
    checkpoint_path = args.relation_classifier_model_path
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found for model '{args.diffusion_version}'")
        print(f"Expected: {checkpoint_path}")
        sys.exit(1)
    
    print(f"Using checkpoint: {checkpoint_path}")
    args.relation_classifier_model_path = checkpoint_path
    
    return args


def generate_image_main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)
    
    # Run generation
    img_generator = init_image_generator(args)
    seed_everything(42, verbose=False)
    img_generator.generate_image(
        prompt=args.prompt,
        seed=-1,
        workspace_path=args.output_dir,
        sample_prefix="test_run",
        save_image=True,
    )
    return img_generator


if __name__ == "__main__":
    args = parse_args()
    generate_image_main(args)
    print("Done")
    