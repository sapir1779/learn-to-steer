import argparse
import json
import os
from tqdm import tqdm
import random


SPATIAL_RELATIONS = [
    "above",
    "below",
    "to the left of",
    "to the right of",
]


def generate_prompt_json_lines(prompts, all_rel_triplets):
    prefixes = [
        # "a top-down view of",
        "a photo of",
        # "Retro Pixel a photo of",
        # "",
        # "a realistic photo of",
        # "A hyper-realistic digital painting of",
        # "A Cinematic sci-fi movie scene of",
        # "a 3D animation of",
    ]
    prompt_lines = []
    for PREFIX in prefixes:
        for prompt, rel_triplets in tqdm(zip(prompts, all_rel_triplets)):
            prompt_line = {
                "tag": "position", 
                "prompt": f"{PREFIX} {prompt}" if len(PREFIX) > 0 else prompt,
                "include": [],
                "version": 2,
            }
            include_dicts = []
            for i, (obj1, rel, obj2) in enumerate(rel_triplets):
                dict1 = {
                    "class": obj2,
                    "count": 1,
                }
                dict2 = {
                    "class": obj1,
                    "count": 1,
                    "position": [rel, i * 2],  ## `i*2` refers to the index in the include list
                }
                include_dicts.append(dict1)
                include_dicts.append(dict2)
            
            prompt_line["include"] = include_dicts
            prompt_lines.append(prompt_line)
    
    return prompt_lines


def get_objects(src):
    if src == "mixed":
        objects_fn = "objects_mixed.txt"
    elif src == "coco":
        objects_fn = "objects_coco.txt"
    elif src == "ood":
        objects_fn = "objects_ood.txt"
    else:
        raise ValueError(f"Unknown src: {src}")
    
    return open(objects_fn, "r").read().splitlines()


def generate_prompt_and_relation_triplets(objects, num_objects, num_relations):
    prompt = ""
    rel_triplets = []
    if num_objects == 2 and num_relations == 1:
        ## 2,1: A r1 B
        obj1, obj2 = random.sample(objects, 2)
        rel1 = random.sample(SPATIAL_RELATIONS, 1)[0]
        prompt = f"a {obj1} {rel1} a {obj2}"
        rel_triplets = [(obj1, rel1, obj2)]
        
    elif num_objects == 3 and num_relations == 2:
        ## 3,2: A r1 B r2 C
        obj1, obj2, obj3 = random.sample(objects, 3)
        rel1, rel2 = random.sample(SPATIAL_RELATIONS, 2)
        prompt = f"a {obj1} {rel1} a {obj2} {rel2} a {obj3}"
        rel_triplets = [(obj1, rel1, obj2), (obj2, rel2, obj3)]
        
    elif num_objects == 4 and num_relations == 2:
        ## 4,2: A r1 B and C r2 D
        obj1, obj2, obj3, obj4 = random.sample(objects, 4)
        rel1, rel2 = random.sample(SPATIAL_RELATIONS, 2)
        prompt = f"a {obj1} {rel1} a {obj2} and a {obj3} {rel2} a {obj4}"
        rel_triplets = [(obj1, rel1, obj2), (obj3, rel2, obj4)]
        
    elif num_objects == 4 and num_relations == 3:
        ## 4,3: A r1 B r2 C r3 D
        obj1, obj2, obj3, obj4 = random.sample(objects, 4)
        rel1, rel2, rel3 = random.sample(SPATIAL_RELATIONS, 3)
        prompt = f"a {obj1} {rel1} a {obj2} {rel2} a {obj3} {rel3} a {obj4}"
        rel_triplets = [(obj1, rel1, obj2), (obj2, rel2, obj3), (obj3, rel3, obj4)]
        
    elif num_objects == 5 and num_relations == 3:
        ## 5,3: A r1 B r2 C and D r3 E
        obj1, obj2, obj3, obj4, obj5 = random.sample(objects, 5)
        rel1, rel2, rel3 = random.sample(SPATIAL_RELATIONS, 3)
        prompt = f"a {obj1} {rel1} a {obj2} {rel2} a {obj3} and a {obj4} {rel3} a {obj5}"
        rel_triplets = [(obj1, rel1, obj2), (obj2, rel2, obj3), (obj4, rel3, obj5)]
        
    else:
        raise ValueError(f"Unsupported combination: {num_objects} objects, {num_relations} relations")
    
    return prompt, rel_triplets
    

def get_mulitple_spatial_relations_prompts(src, num_objects, num_relations, total_prompts, output_folder):
    objects = get_objects(src)
    prompts_set = set()
    prompts = []
    rel_triplets = []
    n = 0
    while n < total_prompts:
        ## Ensure no duplicates
        prompt, rel_triplet = generate_prompt_and_relation_triplets(objects, num_objects, num_relations)
        if prompt in prompts_set:
            continue
        
        prompts.append(prompt)
        prompts_set.add(prompt)
        rel_triplets.append(rel_triplet)
        n += 1
        
    output_path = os.path.join(output_folder, f"position_prompts_V78_{src}_{total_prompts}_{num_objects}objs_{num_relations}rels.jsonl")
    return prompts, output_path, rel_triplets


def dump_prompt_lines(output_path, prompt_lines):
    with open(output_path, "w") as f:
        for prompt_line in prompt_lines:
            f.write(json.dumps(prompt_line) + "\n")
    
    print(f"Saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="mixed", help="Source of objects: mixed / coco / ood")
    parser.add_argument("--num_objects", type=int, default=3, help="Number of objects in the prompt")
    parser.add_argument("--num_relations", type=int, default=2, help="Number of relations in the prompt")
    parser.add_argument("--total_prompts", type=int, default=100, help="Total number of unique prompts to generate")
    parser.add_argument("--output_folder", type=str, 
                        default="/cortex/users/sapiry7/workspace/QUALITATIVE_EXAMPLES__SPATIAL_RELATIONS/MULTIPLE/prompts", 
                        help="Output folder to save the prompts")
    
    args = parser.parse_args()
    args.src = args.src.lower()
    os.makedirs(args.output_folder, exist_ok=True)
    return args


if __name__ == "__main__":
    args = parse_args()
    
    prompts, output_path, rel_triplets = get_mulitple_spatial_relations_prompts(
        args.src, 
        args.num_objects, 
        args.num_relations,
        args.total_prompts, 
        args.output_folder
    )
    prompt_lines = generate_prompt_json_lines(prompts, rel_triplets)
    dump_prompt_lines(output_path, prompt_lines)
    