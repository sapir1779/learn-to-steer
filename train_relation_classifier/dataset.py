from collections import defaultdict
import json
import torch
from torch.utils.data import Dataset


class EnhancedClassifierDataset(Dataset):
    def __init__(self, data_files, rel_name_to_idx_mapping, config, is_train=False, device='cuda'):
        if type(data_files) is not list:
            data_files = [data_files]
        
        self.rel_name_to_idx_mapping = rel_name_to_idx_mapping
        self.config = config
        self.device = device
        self.only_pos = config['training'].get("only_pos", False)
        
        discard_augmented = config['training'].get("discard_augmented", False)
        required_data_size = config['training'].get("data_size", -1)
        reduce_number_of_samples = is_train and required_data_size > 0
        # Counter tracks samples per (prompt_type, gt_relation_name, timestep)
        per_relation_counter = defaultdict(int)
        
        self.data = []
        timesteps = self.config["training"]["timesteps"]
        num_timesteps = self.config["training"]["num_timesteps"]
        need_timestep_check = len(timesteps) < num_timesteps
        
        for data_file in data_files:
            with open(data_file, 'r') as f:
                data_dict = json.load(f)
                
                # Fast path: if no filtering is needed, just extend
                if not need_timestep_check and not discard_augmented and not self.only_pos and not reduce_number_of_samples:
                    self.data.extend(data_dict)
                else:
                    # Slow path: iterate and check only necessary conditions
                    for item in data_dict:
                        if need_timestep_check and item["timestep"] not in timesteps:
                            continue
                        
                        if discard_augmented:
                            item_aug_type = item.get("augmentation_type", "")
                            if len(item_aug_type) > 0:
                                continue
                        
                        if self.only_pos and item["prompt_type"] != "pos":
                            continue
                        
                        if reduce_number_of_samples:
                            gt_relation_name = item["gt_relation_name"]
                            prompt_type = item["prompt_type"]
                            timestep = item["timestep"]
                            counter_key = (prompt_type, gt_relation_name, timestep)
                            if per_relation_counter[counter_key] < required_data_size:
                                self.data.append(item)
                                per_relation_counter[counter_key] += 1
                        else:
                            self.data.append(item)

    def __len__(self):
        return len(self.data)
    
    def get_sample(self, idx):
        return self.data[idx]

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Load the cross-attention maps. Shape = (57, 24, 16, 16)
        map1 = torch.load(sample["cross_attn_maps_paths"][0], weights_only=False)
        map1 = map1.to(torch.float32)
        map2 = torch.load(sample["cross_attn_maps_paths"][1], weights_only=False)
        map2 = map2.to(torch.float32)
        
        # Timestep: Add batch dimension
        timestep = torch.tensor(sample["timestep"], dtype=torch.float32).unsqueeze(0)

        # Ground truth label
        gt_relation_name = sample["gt_relation_name"]
        gt_relation_idx = self.rel_name_to_idx_mapping[gt_relation_name]
        
        maps = torch.stack([map1, map2], dim=0)
        return (maps, timestep), gt_relation_idx
