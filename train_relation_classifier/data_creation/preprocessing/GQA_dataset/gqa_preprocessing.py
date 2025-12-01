from datetime import datetime
import os
import random
import json
import argparse
from collections import defaultdict
import sys
from tqdm import tqdm
from nltk.corpus import wordnet as wn
    
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))


from train_relation_classifier.data_creation.defs import *
from train_relation_classifier.data_creation.preprocessing.GQA_dataset.gqa_dataset import GqaDataset

# Global cache for WordNet is_a lookups
_IS_A_CACHE = {}


def serialize_structure(data, output_path):
    def serialize(obj):
        if isinstance(obj, tuple):
            return str(obj)
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {serialize(k): serialize(v) for k, v in obj.items()}
        else:
            return obj
    
    with open(output_path, "w") as json_file:
        json.dump(serialize(data), json_file, indent=4)

 
def deserialize_structure(input_path):
    def deserialize(obj):
        if isinstance(obj, dict):
            return {
                eval(k) if k.startswith("(") and k.endswith(")") else k: 
                set(v) if isinstance(v, list) else deserialize(v)
                for k, v in obj.items()
            }
        return obj

    with open(input_path, "r") as json_file:
        ds_output = deserialize(json.loads(json_file.read()))
    
    return ds_output


def obj_name_to_all_objects(objs):
    name_to_objs = defaultdict(list)
    for obj in objs:
        name_to_objs[obj.name].append(obj)
    
    return name_to_objs


def is_a(word1, word2):
    """
    Check if word1 is a type of word2 using WordNet hypernym relations (cached).
    """
    cache_key = (word1, word2)
    if cache_key in _IS_A_CACHE:
        return _IS_A_CACHE[cache_key]
    
    synsets1 = wn.synsets(word1, pos='n')
    synsets2 = wn.synsets(word2, pos='n')
    
    result = False
    # Iterate through all synsets of word1
    for syn1 in synsets1:
        if result:
            break
        
        # Get all hypernyms (more general terms) of syn1
        hypernyms = syn1.hypernyms()
        
        # Check if any hypernym matches word2's synsets
        for hypernym in hypernyms:
            if result:
                break
            
            if hypernym in synsets2:
                result = True
                break
            
            # Recursively check higher level hypernyms
            for h in hypernym.hypernyms():
                if h in synsets2:
                    result = True
                    break
    
    _IS_A_CACHE[cache_key] = result
    return result


def has_multiple_instances_of_object(objs):
    """
    Check which two object names depict an "is a" relationship, and mark them both as duplicates.
    """
    dup_objs = set()
    name_to_objs = obj_name_to_all_objects(objs)
    
    # Check which two object names depict a "is a" relationship, and mark them both as duplicates.
    all_object_names = list(name_to_objs.keys())
    for i in range(len(all_object_names)):
        word_i = all_object_names[i]
        if word_i in dup_objs:
            continue
        
        for j in range(len(all_object_names)):
            if i == j:
                continue
            
            word_j = all_object_names[j]
            if is_a(word_i, word_j):
                dup_objs.add(word_i)
                dup_objs.add(word_j)
    
    has_dups = len(dup_objs) > 0
    return has_dups, dup_objs


def reduce_relations_list(flat_relations, max_size, train_val_test_split=None):
    """ 
    Reduces the flat list to a smaller size by extracting ONE relation per image ID.
    If more relations are needed beyond this, randomly samples from the remaining relations.
    
    Args:
        flat_relations: [(img_id, obj1, obj2, rel_name), ...]
        max_size: Target size
        train_val_test_split: dict like {"train": 2000, "val": 300, "test": 200} or None
    
    Returns:
        If train_val_test_split is None: reduced flat list
        Otherwise: dict of {split_type: reduced_flat_list}
    """
    max_size = min(max_size, len(flat_relations))
    
    # Group relations by image ID
    relations_by_img = defaultdict(list)
    for rel in flat_relations:
        img_id = rel[0]
        relations_by_img[img_id].append(rel)
    
    # Extract one relation per image
    output_flat = []
    img_ids = list(relations_by_img.keys())
    shuffled_img_ids = random.sample(img_ids, len(img_ids))
    
    for img_id in shuffled_img_ids:
        if len(output_flat) >= max_size:
            break
        
        # Take first relation from this image
        output_flat.append(relations_by_img[img_id][0])
    
    # If we need more, sample from remaining relations
    if len(output_flat) < max_size:
        needed = max_size - len(output_flat)
        used_set = set(output_flat)
        remaining = [rel for rel in flat_relations if rel not in used_set]
        if len(remaining) > 0:
            sampled = random.sample(remaining, min(needed, len(remaining)))
            output_flat.extend(sampled)
    
    if train_val_test_split is not None:
        output = {}
        current_start_idx = 0
        for split_type, split_size in train_val_test_split.items():
            end_idx = min(current_start_idx + split_size, len(output_flat))
            output[split_type] = output_flat[current_start_idx: end_idx]
            current_start_idx = end_idx
        return output
    else:
        return output_flat


def merge_flat_relations_lists(list_of_flat_lists):
    """Merge multiple flat relation lists into one."""
    merged = []
    for flat_list in list_of_flat_lists:
        merged.extend(flat_list)
    return merged


def create_relation_name_to_index_mapping(output_relations_flat):
    all_relation_names = set()
    for img_id, obj1, obj2, rel_name in output_relations_flat:
        all_relation_names.add(rel_name)
    
    all_relation_names = list(sorted(all_relation_names))
    rel_name_to_idx = {rel_name: str(idx) for idx, rel_name in enumerate(all_relation_names)}
    idx_to_rel_name = {idx: rel_name for idx, rel_name in enumerate(all_relation_names)}
    mapping = {
        "rel_name_to_idx": rel_name_to_idx,
        "idx_to_rel_name": idx_to_rel_name,
    }
    return mapping
        

def dump_output_relations(output_relations_flat, output_folder, max_size=-1, dataset_type="",
                          target_rel_names=None, target_attr_names=None):
    """
    Save flat relations to output folder.
    
    Args:
        output_relations_flat: [(img_id, obj1, obj2, rel_name), ...]
        output_folder: Where to save
        max_size: For naming/info
        dataset_type: "train", "val", or "test"
        target_rel_names: For metadata
        target_attr_names: For metadata
    """
    output_path = ""
    if len(output_folder) > 0:
        size_str = f"size_{max_size}" if max_size > 0 else "ALL"
        new_folder_name = f"{dataset_type}__{size_str}"
        output_path = os.path.join(output_folder, new_folder_name)
        os.makedirs(output_path, exist_ok=True)
        
        # Save flat relations as JSON
        flat_data_path = os.path.join(output_path, FLAT_DATA_FILENAME)
        flat_relations_list = [list(rel) for rel in output_relations_flat]
        with open(flat_data_path, "w") as f:
            json.dump(flat_relations_list, f, indent=4)
        
        # Create and save mapping from unique relation names to indices
        mapping = create_relation_name_to_index_mapping(output_relations_flat)
        mapping_path = os.path.join(output_path, MAPPING_FILENAME)
        with open(mapping_path, "w") as f:
            json.dump(mapping, f, indent=4)
        
        # Save metadata
        result_metadata = {
            FLAT_DATA_PATH_KEY: flat_data_path,
            MAPPING_PATH_KEY: mapping_path,
            DATA_SOURCE_KEY: DATA_SOURCE_GQA,
            SPATIAL_RELATIONS_LIST_KEY: target_rel_names if target_rel_names is not None else [],
            ATTRIBUTES_LIST_KEY: target_attr_names if target_attr_names is not None else [],
        }
        with open(os.path.join(output_path, RESULT_METADATA_FILENAME), "w") as f:
            json.dump(result_metadata, f, indent=4)
        
    return output_path


class PreprocessGqaData:
    """
    Preprocesses the GQA dataset into a list of relations, where each relation is defined by 
    a tuple of (object1, object2) or (object, attribute), and a set of the relation names that correspond to the tuple.
    Note: (Optional) The objects which have multiple instances are not included in the output relations.
    """
    
    def __init__(self, dataset: GqaDataset = None, data_path="", singular_names=True, 
                 skip_mult_instances_relations=True, relation_names_whitelist=SPATIAL_RELATIONS_LIST, 
                 dataset_type="", img_id_to_dataset=None):
        
        if dataset is not None:
            self.dataset = dataset
        elif len(data_path) > 0:
            self.dataset = GqaDataset(data_path, singular_names)
        else:
            self.dataset = None
                    
        self.img_id_to_dataset = self.dataset.data_dict() if self.dataset is not None else img_id_to_dataset
        self.skip_mult_instances_relations = skip_mult_instances_relations
        self.relation_names_whitelist = relation_names_whitelist
        self.dataset_type = dataset_type  ## train/val/all
        
    @staticmethod
    def merge_preprocessors(processor1, processor2):
        merged_img_id_to_dataset = {**processor1.img_id_to_dataset, **processor2.img_id_to_dataset}
        new_processor = PreprocessGqaData(
            skip_mult_instances_relations=processor1.skip_mult_instances_relations,
            relation_names_whitelist=processor1.relation_names_whitelist,
            dataset_type="all",
            img_id_to_dataset=merged_img_id_to_dataset,
        )
        return new_processor
        
        
    def get_relations_from_img_data(self, img_data):
        rels = defaultdict(list)
        id_to_obj = {obj_data.id: obj_data for obj_data in img_data.objects}
        has_dups, dup_objs = has_multiple_instances_of_object(img_data.objects)
        for src_obj_data in img_data.objects:
            if self.skip_mult_instances_relations and (has_dups and src_obj_data.name in dup_objs):
                continue
            
            for rel in src_obj_data.relations:
                if len(self.relation_names_whitelist) == 0 or rel.name not in self.relation_names_whitelist:
                    continue
                
                target_obj_data = id_to_obj[rel.target_object_id]
                if self.skip_mult_instances_relations and \
                    ((has_dups and target_obj_data.name in dup_objs) or src_obj_data.name == target_obj_data.name):
                    continue
                
                rel_triplet = (src_obj_data, rel.name, target_obj_data)
                rels[rel.name].append(rel_triplet)
                
        return rels
    
    
    def generate_all_relations_per_image_data(self, img_data, target_rel_names=None, target_attr_names=None):
        """
        Generates a list of relations for a single image:
        1. Spatial relations: Between two objects. 
            E.g., (img_id, obj1, obj2, "above")
        2. Attribute relations: Object and its attributes.
            E.g., (img_id, obj, attribute, "is")
        
        Returns: [(img_id, obj1, obj2, rel_name), ...]
        """
        output_relations = []
        
        ## Handle attribute relations
        if (target_attr_names is not None) and ((target_rel_names is None) or (RELATION_NAME_IS in target_rel_names)):
            has_mult_instances, dup_objs = has_multiple_instances_of_object(img_data.objects)
            for obj_data in img_data.objects:
                if self.skip_mult_instances_relations and has_mult_instances and (obj_data.name in dup_objs):
                    continue
                
                for attr in obj_data.attributes:
                    if attr not in target_attr_names:
                        continue
                    
                    output_relations.append((img_data.id, obj_data.name, attr, RELATION_NAME_IS))
        
        ## Handle simple relations.
        if (target_rel_names is not None) and (target_attr_names is None): 
            raw_relations = self.get_relations_from_img_data(img_data)
            for rel_triplets in raw_relations.values():
                for obj1, rel_name, obj2 in rel_triplets:
                    if rel_name not in target_rel_names:
                        continue
                        
                    output_relations.append((img_data.id, obj1.name, obj2.name, rel_name))
        
        return output_relations
    
    
    def generate_all_relations(self, target_rel_names=None, target_attr_names=None):
        """
        Returns flat list of all relations: [(img_id, obj1, obj2, rel_name), ...]
        """
        output = []
        for img_id, img_data in self.img_id_to_dataset.items():
            relations = self.generate_all_relations_per_image_data(img_data, target_rel_names, target_attr_names)
            output.extend(relations)
        
        return output
    
    
    def generate_train_val_test_split(self, output_folder, data_splits_per_relation_name_dict, 
                                      target_rel_names=None, target_attr_names=None):
        
        if target_rel_names is None and target_attr_names is None:
            return
        
        output_relations_per_split_type = {split_type: [] for split_type in {"train", "val", "test"}}
        
        # Handle relation names:
        if target_rel_names is not None and len(target_rel_names) > 0:
            for rel_name in tqdm(target_rel_names):
                data_split_dict = data_splits_per_relation_name_dict[rel_name]
                max_size = sum(list(data_split_dict.values()))
                    
                initial_relations = self.generate_all_relations({rel_name}, None)
                reduced_relations = reduce_relations_list(initial_relations, max_size, data_split_dict)
                for split_type, reduced_relations_list in reduced_relations.items():
                    output_relations_per_split_type[split_type].append(reduced_relations_list)
            
        # Handle attributes:
        if target_attr_names is not None and len(target_attr_names) > 0:
            for attr_name in tqdm(target_attr_names):
                attr_data_split_dict = data_splits_per_relation_name_dict[attr_name]
                attr_max_size = sum(list(attr_data_split_dict.values()))
                    
                initial_relations = self.generate_all_relations(None, {attr_name})
                reduced_relations = reduce_relations_list(initial_relations, attr_max_size, attr_data_split_dict)
                for split_type, reduced_relations_list in reduced_relations.items():
                    output_relations_per_split_type[split_type].append(reduced_relations_list)
                
        # Merge and save each split
        output_paths = []
        for split_type, all_reduced_relations in output_relations_per_split_type.items():
            output_relations = merge_flat_relations_lists(all_reduced_relations)
            max_size_dataset_type = sum([v[split_type] for v in data_splits_per_relation_name_dict.values()])
            output_path = dump_output_relations(output_relations, output_folder, max_size_dataset_type, 
                                                dataset_type=split_type,
                                                target_rel_names=target_rel_names,
                                                target_attr_names=target_attr_names)
            output_paths.append(output_path)
            
        return output_relations_per_split_type, output_paths


def create_training_dataset_from_merged_processor(output_folder, train_data_path, val_data_path, exp_name,
                                                  target_rel_names, target_attr_names, 
                                                  data_splits_per_relation_name_dict):
    print(f"Preparing dataset with data splits:\n{data_splits_per_relation_name_dict}")
    
    # Initialize merged processor from train and val processors
    train_processor = PreprocessGqaData(data_path=train_data_path, singular_names=True)
    val_processor = PreprocessGqaData(data_path=val_data_path, singular_names=True)
    merged_processor = PreprocessGqaData.merge_preprocessors(train_processor, val_processor)
        
    print(f"Experiment name: IS_A")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_folder_name = f"{timestamp_str}__IS_A" if len(exp_name) == 0 else f"{timestamp_str}__{exp_name}__IS_A"
    exp_output_folder = os.path.join(output_folder, new_folder_name)
    os.makedirs(exp_output_folder, exist_ok=True)
    
    with open(os.path.join(exp_output_folder, "data_splits_per_relation_name_dict.json"), "w") as f:
        json.dump(data_splits_per_relation_name_dict, f, indent=4)
    
    merged_processor.generate_train_val_test_split(exp_output_folder, data_splits_per_relation_name_dict, 
                                                   target_rel_names, target_attr_names)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess GQA dataset into relation triplets data")
    parser.add_argument('--train_data_path', default=GQA_TRAIN_PATH,
                        help="Path to GQA training data (train_sceneGraphs.json)")
    parser.add_argument('--val_data_path', default=GQA_VAL_PATH,
                        help="Path to GQA validation data (val_sceneGraphs.json)")
    parser.add_argument('--output_folder', default="/cortex/users/sapiry7/workspace/GQA_preprocessing",
                        help="Root folder for output preprocessed data")
    parser.add_argument('--experiment_name', default="",
                        help="Name of the experiment (used for output folder naming)")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    
    target_rel_names = ["above", "below", "to the left of", "to the right of"]
    target_attr_names = ["white", "green", "large", "metal"]
    
    data_splits_per_relation_name_dict = {}
    max_size_per_split = {
        "train": 2000,
        "val": 400,
        "test": 200,
    }
    
    for spatial_rel in target_rel_names:
        splits = { data_split_type: max_size for data_split_type, max_size in max_size_per_split.items() }
        data_splits_per_relation_name_dict[spatial_rel] = splits
    
    for color in target_attr_names:
        splits = {
            data_split_type: max_size // len(target_attr_names) \
                for data_split_type, max_size in max_size_per_split.items()
        }
        data_splits_per_relation_name_dict[color] = splits
    
    create_training_dataset_from_merged_processor(args.output_folder, args.train_data_path, 
                                                  args.val_data_path, args.experiment_name,
                                                  target_rel_names, target_attr_names, 
                                                  data_splits_per_relation_name_dict)
