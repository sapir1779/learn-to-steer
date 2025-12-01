import os
import sys
import torch
import json
import yaml
import argparse
from datetime import datetime
from dataclasses import dataclass
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from train_relation_classifier.dataset import EnhancedClassifierDataset
from train_relation_classifier.model import EnhancedRelationClassifier
from train_relation_classifier.model_utils import save_model
from train_relation_classifier.utils import (
    EarlyStopping, 
    EpochStats, 
    plot_and_save_metrics, 
    copy_latest_early_stopping_start_model
)
from train_relation_classifier.loss_utils import get_loss_criterion
from train_relation_classifier.data_creation.defs import (
    MAPPING_PATH_KEY,
    MAP_SIZE_KEY,
    TARGET_MAP_SIZE_KEY,
    RESULT_DATA_PATH_KEY,
)


@dataclass
class TrainerConfig:
    """Configuration for training the relation classifier."""
    num_epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    device: str
    checkpoint_interval: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    scheduler_enable: bool
    scheduler_factor: float
    scheduler_patience: int
    
    @classmethod
    def from_dict(cls, config: dict) -> 'TrainerConfig':
        training_conf = config['training']
        early_stopping_conf = training_conf['early_stopping']
        scheduler_conf = training_conf['scheduler']
        return cls(
            num_epochs=training_conf['num_epochs'],
            batch_size=training_conf['batch_size'],
            lr=training_conf['lr'],
            weight_decay=training_conf['weight_decay'],
            device=training_conf['device'],
            checkpoint_interval=training_conf['checkpoint_interval'],
            early_stopping_patience=early_stopping_conf['patience'],
            early_stopping_min_delta=early_stopping_conf['min_delta'],
            scheduler_enable=scheduler_conf['enable'],
            scheduler_factor=scheduler_conf['factor'],
            scheduler_patience=scheduler_conf['patience'],
        )


class ConfigProcessor:
    """Handles all config parsing and preprocessing."""
    
    @staticmethod
    def process(config: dict) -> tuple[tuple, dict, dict, str, int]:
        """Process configuration and return all necessary parameters for training."""
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_name = config["training"]["exp_name"]
        
        paths = config['paths']
        dataset_folder_path = paths['dataset_folder_path']
        
        # Override paths if dataset folder is provided
        valid_dataset_folder_path, overriden_paths = handle_dataset_folder_path_arg(dataset_folder_path)
        if valid_dataset_folder_path:
            paths['train_metadata_path'] = overriden_paths[0]
            paths['val_metadata_path'] = overriden_paths[1]
            paths['test_metadata_path'] = overriden_paths[2]
        
        # Create output directory
        checkpoint_dir = paths['checkpoint_dir']
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        output_dir = os.path.join(checkpoint_dir, f"{date_str}_{experiment_name}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Parse metadata files
        train_data_path, train_mapping_path, orig_map_size, target_map_size = parse_result_metadata_file(paths['train_metadata_path'])
        val_data_path = parse_result_metadata_file(paths['val_metadata_path'])[0]
        test_data_path = parse_result_metadata_file(paths['test_metadata_path'])[0]
        data_paths = (train_data_path, val_data_path, test_data_path)
        
        # Update config with metadata
        config['training']["orig_map_size"] = int(orig_map_size)
        if target_map_size > 0:
            config['model']["map_size"] = target_map_size
        
        # Get relation mappings
        rel_name_to_idx, idx_to_rel_name = parse_relation_index_mapping_file(train_mapping_path)
        
        # Dump config to output path
        with open(os.path.join(output_dir, "config.yaml"), "w") as file:
            yaml.dump(config, file, default_flow_style=False)
        
        print(yaml.dump(config, sort_keys=True, default_flow_style=False))
        
        input_channels = 57 * 24
        
        return data_paths, rel_name_to_idx, idx_to_rel_name, output_dir, input_channels


class RelationClassifierTrainer:
    """Encapsulates model initialization, training, and evaluation logic."""
    
    def __init__(self, config: dict, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, 
                 rel_name_to_idx: dict, idx_to_rel_name: dict, output_dir: str, gpu_indices: list, input_channels: int):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.rel_name_to_idx = rel_name_to_idx
        self.idx_to_rel_name = idx_to_rel_name
        self.output_dir = output_dir
        self.gpu_indices = gpu_indices
        self.input_channels = input_channels
        
        # Extract trainer configuration
        self.trainer_config = TrainerConfig.from_dict(config)
        self.relation_names = sorted(rel_name_to_idx.keys())
        
        # Initialize model, optimizer, and scheduler
        self._init_model()
        self._init_optimizer()
        self._init_scheduler()
        
        # Initialize loss criterion
        self.loss_criterion = get_loss_criterion()
        
        # Save model parameters
        self._dump_model_params()
    
    def _init_model(self) -> None:
        model_config = self.config["model"]
        training_config_dict = self.config["training"]
        device = self.trainer_config.device
        self.model = EnhancedRelationClassifier(self.rel_name_to_idx, self.idx_to_rel_name, 
                                                 self.input_channels, model_config, training_config_dict)
        
        # Wrap in DataParallel if multiple GPUs
        if len(self.gpu_indices) > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(device)
        
    
    def _init_optimizer(self) -> None:
        lr = self.trainer_config.lr
        print(f"Learning rate: {lr}")
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, 
                                      weight_decay=self.trainer_config.weight_decay)
    
    def _init_scheduler(self) -> None:
        self.scheduler = None
        if self.trainer_config.scheduler_enable:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", 
                factor=self.trainer_config.scheduler_factor, 
                patience=self.trainer_config.scheduler_patience, 
                min_lr=1e-6, threshold=0.0
            )
    
    def _dump_model_params(self) -> None:
        """Dump model parameter count to file."""
        model_input = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        num_params = sum(p.numel() for p in model_input.parameters())
        print(f"Total number of parameters: {num_params}")
        if self.output_dir is not None:
            with open(os.path.join(self.output_dir, "model_num_parameters.txt"), "w") as f:
                f.write(str(num_params))
    
    def save_model(self) -> None:
        if self.output_dir is not None:
            save_model(self.model, os.path.join(self.output_dir, "final_model.pth"))
            copy_latest_early_stopping_start_model(self.output_dir)
            
    
    def train(self) -> tuple[list, list, list, list, list, list]:
        """Run the training loop and return losses and accuracies."""
        print("Start training")
        
        train_losses = []
        val_losses = []
        test_losses = []
        train_accuracies = []
        val_accuracies = []
        test_accuracies = []
        
        early_stopping = EarlyStopping(patience=self.trainer_config.early_stopping_patience, 
                                       min_delta=self.trainer_config.early_stopping_min_delta)
        
        for epoch in range(self.trainer_config.num_epochs):
            self.model.train()
            stats = EpochStats("Train", epoch, self.trainer_config.num_epochs, num_batches=len(self.train_loader))
            for (maps, timestep), labels in tqdm(self.train_loader):
                # Move inputs and labels to the same device as the model
                maps, timestep = maps.to(self.trainer_config.device), timestep.to(self.trainer_config.device)
                labels = labels.to(self.trainer_config.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(maps, timestep)
                loss = self.loss_criterion(outputs, labels)
                
                # Track training accuracy
                stats.update_batch(outputs, labels, timestep, loss)
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
            
            # Compute and store metrics
            epoch_train_loss, train_accuracy = stats.process_epoch(self.relation_names, self.output_dir, self.model)
            train_losses.append(epoch_train_loss)
            train_accuracies.append(train_accuracy)
            
            # Validation
            val_loss, val_accuracy = self.evaluate(self.val_loader, src="Val", epoch=epoch, 
                                                    num_epochs=self.trainer_config.num_epochs)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # Checkpoint saving
            if self.output_dir is not None and (epoch+1) % self.trainer_config.checkpoint_interval == 0:
                save_model(self.model, os.path.join(self.output_dir, f"model__epoch_{epoch}.pth"))
            
            # Early stopping check
            early_stop, started_no_improvement = early_stopping.should_stop(val_loss)
            if started_no_improvement and self.output_dir is not None:
                save_model(self.model, os.path.join(self.output_dir, f"model__epoch_{epoch}__early_stopping_start.pth"))
            
            if early_stop and self.output_dir is not None:
                save_model(self.model, os.path.join(self.output_dir, f"model__epoch_{epoch}__early_stopping.pth"))
                break
        
        # Save final plots and metrics
        plot_and_save_metrics(train_losses, val_losses, test_losses, 
                              train_accuracies, val_accuracies, test_accuracies, 
                              self.output_dir)
        return train_losses, val_losses, test_losses, train_accuracies, val_accuracies, test_accuracies


    def evaluate(self, data_loader: DataLoader, src: str = "Val", epoch: int = 0, num_epochs: int = 1) -> tuple[float, float]:
        """Evaluate model on a given data loader."""
        stats = EpochStats(src, epoch, num_epochs, num_batches=len(data_loader))
        self.model.eval()
        with torch.no_grad():
            for (maps, timestep), labels in data_loader:
                # Move inputs and labels to the same device
                maps, timestep = maps.to(self.trainer_config.device), timestep.to(self.trainer_config.device)
                labels = labels.to(self.trainer_config.device)
                
                # Forward pass
                outputs = self.model(maps, timestep)
                loss = self.loss_criterion(outputs, labels)
                
                # Update stats
                stats.update_batch(outputs, labels, timestep, loss)
        
        epoch_loss, accuracy = stats.process_epoch(self.relation_names, self.output_dir, self.model)
        return epoch_loss, accuracy


def parse_relation_index_mapping_file(mapping_path: str) -> tuple[dict, dict]:
    with open(mapping_path, "r") as json_file:
        mapping = json.load(json_file)
        
    rel_name_to_idx = mapping["rel_name_to_idx"]
    rel_name_to_idx = {k: int(v) for k, v in rel_name_to_idx.items()}
    idx_to_rel_name = mapping["idx_to_rel_name"]
    idx_to_rel_name = {int(k): v for k, v in idx_to_rel_name.items()}
    return rel_name_to_idx, idx_to_rel_name


def parse_result_metadata_file(metadata_files: str | list) -> tuple[list, str, int, int]:
    if type(metadata_files) is not list:
        metadata_files = [metadata_files]
        
    data_paths, mapping_paths, orig_map_sizes, target_sizes = [], [], [], []
    for metadata_file in metadata_files:
        with open(metadata_file, "r") as json_file:
            result_metadata = json.load(json_file)
            
        data_path = result_metadata[RESULT_DATA_PATH_KEY]
        mapping_path = result_metadata[MAPPING_PATH_KEY]
        orig_map_size = result_metadata[MAP_SIZE_KEY] if MAP_SIZE_KEY in result_metadata else -1
        target_map_size = result_metadata[TARGET_MAP_SIZE_KEY] if TARGET_MAP_SIZE_KEY in result_metadata else -1
        data_paths.append(data_path)
        mapping_paths.append(mapping_path)
        orig_map_sizes.append(orig_map_size)
        target_sizes.append(target_map_size)
    
    mapping_path = mapping_paths[0]
    orig_map_size = int(orig_map_sizes[0])
    target_map_size = int(target_sizes[0])
    return data_paths, mapping_path, orig_map_size, target_map_size


def handle_dataset_folder_path_arg(dataset_folder_path: str | list) -> tuple[bool, tuple[list, list, list]]:
    # Overrides the given train/val/test direct "result_metadata.json" paths
    if type(dataset_folder_path) is not list:
        dataset_folder_path = [dataset_folder_path]
    
    train_paths, val_paths, test_paths = [], [], []
    valid = False
    for i in range(len(dataset_folder_path)):
        current_path = dataset_folder_path[i]
        if len(current_path) > 0:
            valid = True
            for folder_name in os.listdir(current_path):
                folder_path = os.path.join(current_path, folder_name)
                if "train" in folder_name:
                    train_paths.append(os.path.join(folder_path, "result_metadata.json"))
                elif "val" in folder_name:
                    val_paths.append(os.path.join(folder_path, "result_metadata.json"))
                elif "test" in folder_name:
                    test_paths.append(os.path.join(folder_path, "result_metadata.json"))
    
    return valid, (train_paths, val_paths, test_paths)


def create_data_loaders(train_data_path: str, val_data_path: str, test_data_path: str, config: dict, 
                        rel_name_to_idx: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    print("Create train dataset")
    batch_size = config['training']['batch_size']
    train_dataset = EnhancedClassifierDataset(train_data_path, rel_name_to_idx, config, is_train=True)
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size,
                              shuffle=config['training']['shuffle'])
    
    print("Create validation dataset")
    val_dataset = EnhancedClassifierDataset(val_data_path, rel_name_to_idx, config)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    print("Create test dataset")
    test_dataset = EnhancedClassifierDataset(test_data_path, rel_name_to_idx, config)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    return train_loader, val_loader, test_loader


def run(args: argparse.Namespace, config: dict, gpu_indices: list) -> tuple[tuple, float, float, str]:
    ## Process config and input paths
    data_paths, rel_name_to_idx, idx_to_rel_name, output_dir, input_channels = ConfigProcessor.process(config)
    
    # Dump the input args
    if args is not None:
        with open(os.path.join(output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
    
    ## Create datasets and loaders: train, validation, and test
    train_data_path, val_data_path, test_data_path = data_paths
    train_loader, val_loader, test_loader = create_data_loaders(train_data_path, val_data_path, test_data_path, 
                                                                config, rel_name_to_idx)

    ## Train and evaluate
    trainer = RelationClassifierTrainer(config, train_loader, val_loader, test_loader,
                                       rel_name_to_idx, idx_to_rel_name, output_dir, 
                                       gpu_indices, input_channels)
    
    train_output = trainer.train()
    test_loss, test_accuracy = trainer.evaluate(trainer.test_loader, src="Test", epoch=0, num_epochs=1)

    ## Save the model
    trainer.save_model()
        
    return train_output, test_loss, test_accuracy, output_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', 
                        default="train_relation_classifier/configs/flux_schnell_config.yaml", 
                        help="Path to training config YAML file")
    parser.add_argument('--train_metadata_path', default="", help="Path to train metadata JSON file")
    parser.add_argument('--val_metadata_path', default="", help="Path to validation metadata JSON file")
    parser.add_argument('--test_metadata_path', default="", help="Path to test metadata JSON file")
    parser.add_argument('--dataset_folder_path', nargs='+', type=str, default="", 
                        help="Path to cross-attnention dataset folder(s) containing train/val/test "
                             "subfolders with result_metadata.json files")
    parser.add_argument('--checkpoint_dir', default="", help="Path to checkpoint directory")
    parser.add_argument('--data_size', default=-1, help="Size of the dataset to use for training (for debugging)")
    parser.add_argument('--gpu_indices', default=[], type=int, nargs="+", help="List of available GPU indices")
    parser.add_argument('--timesteps', default=[], type=int, nargs="+", 
                        help="Timesteps to use for training (optional, in combination with config file)")
    parser.add_argument('--map_size', default="", help="Cross-attention HxW map size")
    parser.add_argument('--experiment_name', default="", help="The name of the current experiment")
    args = parser.parse_args()
    print(f"{args=}")
    return args


def update_config(args: argparse.Namespace, config: dict) -> dict:
    # Update train/val/test metadata paths, if applicable
    if len(args.train_metadata_path) > 0:
        config['paths']["train_metadata_path"] = args.train_metadata_path
    if len(args.val_metadata_path) > 0:
        config['paths']["val_metadata_path"] = args.val_metadata_path
    if len(args.test_metadata_path) > 0:
        config['paths']["test_metadata_path"] = args.test_metadata_path
        
    if len(args.dataset_folder_path) > 0:
        config['paths']["dataset_folder_path"] = args.dataset_folder_path
        
    if len(args.checkpoint_dir) > 0:
        config['paths']["checkpoint_dir"] = args.checkpoint_dir
    
    if len(args.timesteps) > 0:
        config['training']["timesteps"] = [int(timestep) for timestep in args.timesteps]
    
    if len(args.experiment_name) > 0:
        config['training']["exp_name"] = args.experiment_name
        
    if len(args.map_size) > 0:
        config['model']["map_size"] = int(args.map_size)
    
    if int(args.data_size) > 0:
        config['training']["data_size"] = int(args.data_size)
        
    return config


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu_idx) for gpu_idx in args.gpu_indices])

    config_path = args.config_path
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
      
    config = update_config(args, config)
    
    run(args, config, args.gpu_indices)
