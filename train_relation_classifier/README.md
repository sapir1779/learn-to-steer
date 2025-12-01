# Overview
This project implements a two-phase pipeline for training a spatial relation classifier:

1. **Phase 1 - Relation Triplet Extraction** (Optional): Parse existing datasets (GQA, ORM) to create structured relation samples
2. **Phase 2 - Cross-Attention Map Generation**: Use diffusion model inversion to generate training data with cross-attention maps

The classifier takes as input:
- Cross-attention maps from object 1 (all heads and layers)
- Cross-attention maps from object 2 (all heads and layers)  
- Timestep information

And predicts the spatial relation (e.g., "above", "to the left of", etc.).


## Training Data Generation

### Phase 1: Relation Triplet Extraction (Optional)
**Location:** `train_relation_classifier/data_creation/preprocessing/`

Extracts structured relation samples `{img_id, obj1, obj2, relation}` from raw image datasets:
- **GQA**: Scene graph-based visual question answering dataset  Source: [https://arxiv.org/pdf/1902.09506]
- **ORM**: Spatially-aligned `{image, prompt}` pairs. Source: [https://arxiv.org/pdf/2501.13926]

Output: JSON files with relation triplets and mappings from relation name to an index (and vice versa).

Uploaded triplets: [https://drive.google.com/drive/folders/1FjRsnSUJIgn9yDAZUNl8_6CnGpv-rP-T?usp=sharing]


```bash
cd train_relation_classifier/data_creation/preprocessing/GQA_dataset
python gqa_preprocessing.py \
  --train_data_path /path/to/gqa/train_sceneGraphs.json \
  --val_data_path /path/to/gqa/val_sceneGraphs.json \
  --output_folder /path/to/output \
  --experiment_name my_experiment
```

This creates relation triplets for the target relations: "above", "below", "to the left of", "to the right of".


### Phase 2: Cross-Attention Maps Generation
**Location:** `train_relation_classifier/data_creation/attention_maps_generation/`

The main pipeline that:
1. Takes relation triplets from Phase 1
2. Generates prompts for each triplet
3. Performs denoising model inversion to obtain cross-attention maps
4. Saves cross-attention tensors and metadata for training

Output: Training datasets with cross-attention maps ready for classifier training.

Each sample in the training data contains:
```python
{
  "flux_output_path": str,              # path to inverted image and attention maps
  "prompt": str,                        # generated prompt for the relation triplet
  "timestep": int,                      # inversion timestep
  "cross_attn_maps_paths": list,        # paths to cross-attention map tensors for [obj1, obj2]
  "rel_objects": [str, str],            # object names [obj1, obj2]
  "gt_relation_name": str,              # ground truth relation (e.g., "above", "to the left of")
  "img_id": str,                        # image ID from source dataset
  "prompt_type": str                    # type of prompt (e.g., "pos", "neg")
}
```

Examples of training data: [https://drive.google.com/drive/folders/1Ua5RMsHBWoTNKC5x0ZERZ7gZZLCeNvYk?usp=sharing]

Prerequisites: Download the image datasets of either GQA or ORM through their official channels.

```bash
python train_relation_classifier/data_creation/attention_maps_generation/generate_relations_data.py \
  --metadata_path path/to/metadata/from/phase1.json \  ## "result_metadata.json"
  --diffusion_version schnell \
  --output_path ./generated_data \
  --gpu_indices 0 1 2 \
  --gqa_images_root /path/to/gqa/images \
  --orm_images_root /path/to/orm/images \
  --target_timesteps 1 3
```
This creates cross-attention maps and timesteps training data.

Can also check out `create_data_schnell.sh`


## Training the Classifier
```bash
python train_relation_classifier/train.py \
  --config_path path/to/config.yaml \
  --dataset_folder_path path/cross_attn_maps/data \
  --checkpoint_dir path/to/checkpoints \
  --gpu_indices 0 1 2 3
```

Key arguments:
- `--dataset_folder_path`: Directory containing train/val/test splits
- `--checkpoint_dir`: Where to save model checkpoints


## Configuration
Edit `train_relation_classifier/configs/flux_schnell_config.yaml` to customize:
- Training hyperparameters (batch size, learning rate, epochs)
- Timesteps to train on
- Data paths


**Available Models**:
- `schnell`: FLUX.1-schenll
- `dev`: FLUX.1-dev
- `sd14`: Stable Diffusion 1.4
- `sd2`: Stable Diffusion 2.1
