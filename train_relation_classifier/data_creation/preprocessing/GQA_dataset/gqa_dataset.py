import json
from dataclasses import dataclass
import os
from typing import List
import inflect
from PIL import Image

from train_relation_classifier.data_creation.defs import GQA_IMAGES_PATH


@dataclass
class RelationData:
    name: str
    target_object_id: str
    
@dataclass
class ObjectData:
    id: str
    name: str
    x: int
    y: int
    w: int
    h: int
    attributes: List[str]
    relations: List[RelationData]
    count: int
    is_plural: bool

@dataclass
class ImageData:
    id: str
    width: int
    height: int
    objects: List[ObjectData]
    scene_data: dict
    

class GqaDataset:
    def __init__(self, data_path, singular_names=False):
        self.singular_names = singular_names
        self.inflect_engine = inflect.engine()
        self.dataset = self.parse_dataset(data_path)
        self.img_id_to_dataset = {img_data.id: img_data for img_data in self.dataset}
        
    def plural_to_singular(self, word):
        """
        Returns the singular form if conversion succeeds, otherwise returns the original word, 
        and whether the word is a plural or singular noun.
        """
        
        if word.endswith("ss"):
            return True, word
        
        if word == "bus":
            return False, word
        
        if word == "buses":
            return True, word
        
        singular_word = self.inflect_engine.singular_noun(word)
        if singular_word is not False:
            return True, singular_word
        else:
            return False, word
    
        
    def parse_dataset(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)

        dataset = []
        for img_id, scene_data in data.items():
            height = scene_data["height"]
            width = scene_data["width"]
            objects_data = []
            for obj_id, obj_data in scene_data["objects"].items():
                obj_name = obj_data["name"]
                is_plural, singular_obj_name = self.plural_to_singular(obj_name)
                if self.singular_names:
                    obj_name = singular_obj_name
                
                x = obj_data["x"]
                y = obj_data["y"]
                w = obj_data["w"]
                h = obj_data["h"]
                attributes = obj_data["attributes"]
                relations_data = [RelationData(rel_data["name"], rel_data["object"]) for rel_data in obj_data["relations"]]
                count = obj_data["count"] if "count" in obj_data else -1
                objects_data.append(ObjectData(obj_id, obj_name, x, y, w, h, attributes, relations_data, count, is_plural))
            
            img_data = ImageData(img_id, width, height, objects_data, scene_data)
            # img_data = ImageData(img_id, width, height, objects_data)
            dataset.append(img_data)
        
        return dataset
    
    def data_list(self):
        return self.dataset
    
    def data_dict(self):
        return self.img_id_to_dataset


def load_gqa_image(img_id, root_dir=GQA_IMAGES_PATH):
    image = Image.open(os.path.join(root_dir, f"{img_id}.jpg"))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image
    