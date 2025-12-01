import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

from train_relation_classifier.model import EnhancedRelationClassifier


def save_model(model, output_path):
    if isinstance(model, torch.nn.DataParallel):
        model_input = model.module
    else:
        model_input = model
    
    model_state_dict = model_input.state_dict()
    rel_name_to_idx = model_input.rel_name_to_idx
    idx_to_rel_name = model_input.idx_to_rel_name
    input_channels = model_input.input_channels
    config = model_input.config
    training_config = model_input.training_config
    
    output_dict = {
        'model_state_dict': model_state_dict,
        'rel_name_to_idx': rel_name_to_idx,
        'idx_to_rel_name': idx_to_rel_name,
        'input_channels': input_channels,
        'config': config,
        'training_config': training_config,
    }
    torch.save(output_dict, output_path)


def load_model(model_path, device_name="cuda"):
    device = torch.device(device_name)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    rel_name_to_idx = checkpoint['rel_name_to_idx']
    idx_to_rel_name = checkpoint['idx_to_rel_name']
    input_channels = checkpoint['input_channels'] if "input_channels" in checkpoint else 57 * 24
    model_config = checkpoint["config"] if "config" in checkpoint else None
    training_config = checkpoint["training_config"] if "training_config" in checkpoint else None
    
    if model_config is None:
        raise ValueError("Model config not found in checkpoint.")
    
    model = EnhancedRelationClassifier(rel_name_to_idx, idx_to_rel_name, input_channels, 
                                       model_config, training_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
