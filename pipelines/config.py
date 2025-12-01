import json
import os


class Config:

    DEFAULT_CONFIG = {
        ## Test-time optimization config
        'test_time_optimization': {
            'enable': False,
            'start_idx': 0,
            'end_idx': 1,
            'step_sizes': [],
            'min_loss': 0,
            'iterative_refinement': {
                'enable': False,
                'loss_targets_per_iteration': {},
                'max_refinement_steps': 10,
            },
            "test_classification": False,
        },

        ## Cross-attention config
        'attention': {
            'enable': True,
            'target_timesteps': [],
            'store_heads': True,
            'remove_start_token': False,
            'include_color': True,
            'dump_sources': ['head', 'layer', 'block', 'overall_mean'],
            'map_size': 16,
        },

        ## General config
        'debug': True,
        'decode_all_latents': False,
        'workspace_path': "/cortex/users/sapiry7/workspace",
        'sample_prefix': "",
        'use_timestamp': True,
        'num_inference_steps': 4,
    }

    def __init__(self, config_dict=DEFAULT_CONFIG):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

    def to_dict(self):
        # Convert the Config object back to a dictionary for serialization
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                value = value.to_dict()
            result[key] = value
        return result
    
    def dump(self, output_path, filename="config.json"):
        config_file = os.path.join(output_path, filename)
        with open(config_file, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    def __str__(self):
        return f'{json.dumps(self.to_dict(), indent=4)}'

    def __repr__(self):
        return self.__str__()
