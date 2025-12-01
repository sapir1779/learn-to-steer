"""
ORM dataset preprocessing for relation classification training data generation.
Taken from: the paper: "Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step"
[https://arxiv.org/pdf/2501.13926] [https://huggingface.co/datasets/ZiyuG/Image-Generation-CoT/tree/main]
"""
import argparse
from datetime import datetime
import os
import json
import re
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

from prompt_parsing_utils import PromptParser
from train_relation_classifier.data_creation.defs import (
    FLAT_DATA_FILENAME, 
    MAPPING_FILENAME, 
    RESULT_METADATA_FILENAME,
    FLAT_DATA_PATH_KEY,
    MAPPING_PATH_KEY,
    DATA_SOURCE_KEY,
    DATA_SOURCE_ORM,
    ORM_IMAGES_PATH,
    RELATION_NAME_IS,
    SPATIAL_RELATIONS_LIST_KEY,
    ATTRIBUTES_LIST_KEY,
)
from train_relation_classifier.data_creation.preprocessing.GQA_dataset.gqa_preprocessing import (
    serialize_structure, 
    create_relation_name_to_index_mapping
)


def convert_to_standard_relation_name(input_relation):
    if input_relation == "left of":
        output_relation = "to the left of"
    elif input_relation == "right of":
        output_relation = "to the right of"
    else:
        output_relation = input_relation
        
    return output_relation


def reduce_relations_output(relations_output, max_size):
    max_size = min(max_size, len(relations_output)) if max_size > 0 else len(relations_output)
    
    # Add unique prompts first
    done_triplets = set()
    final_output = set()
    for quadruplet in relations_output:
        _, obj1, obj2, relation = quadruplet[:4]
        triplet = (obj1, obj2, relation)
        if triplet in done_triplets:
            continue
        
        if len(final_output) >= max_size:
            break
        
        done_triplets.add(triplet)
        final_output.add(quadruplet)
    
    # Add the remaining elements, if didn't reach the max size
    if len(final_output) < max_size:
        for quadruplet in relations_output:
            if quadruplet in final_output:
                continue
            
            if len(final_output) >= max_size:
                break
            
            final_output.add(quadruplet)
    
    final_output = list(final_output)
    return final_output


def prepare_train_val_test_splits(output_per_relation, data_splits_per_relation_name_dict):
    data_splits = defaultdict(list)
    data_sizes = defaultdict(list)
    for relation_name, relations_output in output_per_relation.items():
        relation_data_splits = data_splits_per_relation_name_dict[relation_name]
        max_size = sum(list(relation_data_splits.values()))
        reduced_relations = reduce_relations_output(relations_output, max_size)
        total = len(reduced_relations)
        
        split_start_idx = 0
        for split_type, split_size in relation_data_splits.items():
            split_end_idx = min(split_start_idx + split_size, total)
            split_relations = reduced_relations[split_start_idx: split_end_idx]
            
            data_splits[split_type] += split_relations
            data_sizes[split_type].append(len(split_relations))
            
            split_start_idx = min(split_end_idx, total-1)
        
    return data_splits, data_sizes


def dump_data(output_folder, data_splits, data_sizes, spatial_keywords, color_keywords):
    for dataset_type, output_relations in data_splits.items():  
        size = sum(data_sizes[dataset_type])      
        new_folder_name = f"{dataset_type}__size_{size}"
        output_path = os.path.join(output_folder, new_folder_name)
        os.makedirs(output_path, exist_ok=True)
    
        # Flatten image ID to relations dictionary and dump it
        flat_data_path = os.path.join(output_path, FLAT_DATA_FILENAME)
        serialize_structure(output_relations, flat_data_path)
        
        # Create mapping from unique relation name to index (and vice versa), and dump it
        mapping_path = os.path.join(output_path, MAPPING_FILENAME)
        rel_name_idx_mapping_dict = create_relation_name_to_index_mapping(output_relations)
        with open(mapping_path, "w") as f:
            json.dump(rel_name_idx_mapping_dict, f, indent=4)
            
        result_metadata = {
            FLAT_DATA_PATH_KEY: flat_data_path,
            MAPPING_PATH_KEY: mapping_path,
            DATA_SOURCE_KEY: DATA_SOURCE_ORM,
            SPATIAL_RELATIONS_LIST_KEY: spatial_keywords,
            ATTRIBUTES_LIST_KEY: color_keywords,
        }
        with open(os.path.join(output_path, RESULT_METADATA_FILENAME), "w") as f:
            json.dump(result_metadata, f, indent=4)
        

def generate_training_data(dataset_json_path, spatial_keywords, color_keywords, 
                           data_splits_per_relation_name_dict, output_folder):    
    with open(dataset_json_path, "r") as f:
        data = json.load(f)

    # Extract only positive samples, i.e., when the answer is "yes".
    output_per_relation = defaultdict(list)
    prompt_parser = PromptParser(skip_attrs=False)
    for item in tqdm(data):
        image_path = item["image"]
        answer = item["conversations"][-1]["value"].strip().lower()
        if answer == "yes":
            question = item["conversations"][0]["value"]
            match = re.search(r"prompt: (.*)\. Does", question)
            if match:
                prompt = match.group(1)
                for relation in spatial_keywords:
                    if relation in prompt:
                        relation_triplets = prompt_parser.extract_relation_triplets_from_prompt(prompt)
                        if len(relation_triplets) == 0:
                            continue
                        
                        for obj1, parsed_relation, obj2 in relation_triplets:
                            output_relation = convert_to_standard_relation_name(relation)
                            if parsed_relation != output_relation:
                                continue
                            output_item = (image_path, obj1, obj2, output_relation)
                            output_per_relation[relation].append(output_item)
                            break
                    
                for color in color_keywords:
                    if color in prompt:
                        relation_triplets = prompt_parser.extract_relation_triplets_from_prompt(prompt)
                        if len(relation_triplets) == 0:
                            continue
                        
                        for j in range(len(relation_triplets)):
                            obj, parsed_relation, attr = relation_triplets[j]
                            if color != attr or parsed_relation != RELATION_NAME_IS \
                                or (attr not in color_keywords):
                                continue
                            
                            output_relation = convert_to_standard_relation_name(parsed_relation)
                            output_item = (image_path, obj, attr, output_relation)
                            output_per_relation[f"{output_relation}_{attr}"].append(output_item)
                        break
            else:
                print(f"No match for {question}")
    
    data_splits, data_sizes = prepare_train_val_test_splits(output_per_relation, 
                                                            data_splits_per_relation_name_dict)
    dump_data(output_folder, data_splits, data_sizes, spatial_keywords, color_keywords)


def load_orm_image(img_path, root_dir=ORM_IMAGES_PATH):
    image = Image.open(os.path.join(root_dir, f"{img_path}"))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess ORM dataset into relation triplets data")
    parser.add_argument('--data_path', required=True, default="/cortex/users/sapiry7/workspace/ORM_GEN_EVAL_DATA/orm.json",
                        help="Path to ORM dataset JSON file (orm.json)")
    parser.add_argument('--output_folder', default="/cortex/users/sapiry7/workspace/ORM_preprocessing",
                        help="Root folder for output preprocessed data")
    parser.add_argument('--experiment_name', default="",
                        help="Name of the experiment (used for output folder naming)")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    
    spatial_keywords = ["above", "below", "left of", "right of"]
    color_keywords = ["white", "green"]
    
    data_splits_per_relation_name_dict = {}
    max_size_per_split = {
        "train": 800,
        "val": 100,
        "test": 70,
    }

    for spatial_rel in spatial_keywords:
        splits = { data_split_type: max_size for data_split_type, max_size in max_size_per_split.items() }
        data_splits_per_relation_name_dict[spatial_rel] = splits
    
    for color in color_keywords:
        splits = {
            data_split_type: max_size // len(color_keywords) \
                for data_split_type, max_size in max_size_per_split.items()
        }
        data_splits_per_relation_name_dict[f"{RELATION_NAME_IS}_{color}"] = splits

    ## Create output folder
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_folder_name = f"{timestamp_str}__{args.experiment_name}" if len(args.experiment_name) > 0 else f"{timestamp_str}"
    exp_output_folder = os.path.join(args.output_folder, new_folder_name)
    os.makedirs(exp_output_folder, exist_ok=True)

    with open(os.path.join(exp_output_folder, "data_splits_per_relation_name_dict.json"), "w") as f:
        json.dump(data_splits_per_relation_name_dict, f, indent=4)

    generate_training_data(args.data_path, spatial_keywords, color_keywords, 
                           data_splits_per_relation_name_dict, 
                           output_folder=exp_output_folder)
