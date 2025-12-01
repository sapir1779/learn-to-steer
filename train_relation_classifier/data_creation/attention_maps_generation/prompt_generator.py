from typing import List
import numpy as np
from abc import ABC, abstractmethod

from train_relation_classifier.data_creation.defs import RELATION_NAME_IS


POS_PROMPT_TYPE = "pos"
NEG_PROMPT_TYPE = "neg"


def sample_relations_from_set(relation_names_set, num_relations):
    options = list(relation_names_set)
    output_relations = np.random.choice(options, size=num_relations, replace=False)
    return output_relations


class PromptGenerator(ABC):
    def __init__(self, all_relation_names, num_prompts):
        self.all_relation_names = all_relation_names
        self.num_prompts = num_prompts
    
    @abstractmethod
    def generate_source_prompts(self, obj1, obj2, rel_name) -> List[str]:
        pass
        
    def generate_prompt(self, obj1, obj2, input_rel_name, output_rel_name):
        """
        When we generate a negative prompt, we must preserve the structure of the original relation.
        Meaning that for spatial relations, the prompt format is: "OBJ1 [spatial relation] OBJ2".
        For the "is a" relation, the format is: "ATTR OBJ".
        """
        if input_rel_name == RELATION_NAME_IS:
            obj, attr = obj1, obj2
            if input_rel_name == output_rel_name:
                prompt = f"A {attr} {obj}"
            else:
                prompt = f"A {attr} {output_rel_name} a {obj}"
        else:
            prompt = f"A {obj1} {output_rel_name} a {obj2}"
            
        return prompt
    

class PosNegPromptGenerator(PromptGenerator):
    def __init__(self, all_relation_names, num_prompts):
        super().__init__(all_relation_names, num_prompts)

    def sample_negative_relations(self, rel_name, num_relations):
        # Exclude the input relation (rel_name) from the list of possible negative relations
        neg_relation_names_set = self.all_relation_names.difference({rel_name})
        return sample_relations_from_set(neg_relation_names_set, num_relations)
    
    def generate_source_prompts(self, obj1, obj2, rel_name):
        src_prompts = []
        
        # Generate positive prompt
        pos_prompt = self.generate_prompt(obj1, obj2, rel_name, rel_name)
        src_prompts.append((pos_prompt, POS_PROMPT_TYPE, rel_name, [obj1, obj2]))
        
        # Generate negative prompts
        neg_rel_names = self.sample_negative_relations(rel_name, self.num_prompts)
        for neg_rel_name in neg_rel_names:
            neg_prompt = self.generate_prompt(obj1, obj2, rel_name, neg_rel_name)
            src_prompts.append((neg_prompt, NEG_PROMPT_TYPE, rel_name, [obj1, obj2]))
        
        return src_prompts
