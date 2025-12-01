import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import sys
import time
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from summary_scores import dump_metrics
import warnings
warnings.filterwarnings("ignore")


POSITION_THRESHOLD = 0.1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("imagedir", type=str)
    parser.add_argument("--outfile", type=str, default="")
    parser.add_argument('--gpu_indices', default=[0], type=int, nargs="+", help="List of available GPU indices")
    
    args = parser.parse_args()
    print(f"{args.gpu_indices=}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu_idx) for gpu_idx in args.gpu_indices])
    
    args.outfile = os.path.join(args.imagedir, "results.jsonl")
    # args.outfile = os.path.join(args.imagedir, "results__new.jsonl")
    print(f"{args=}")
    return args


def timed(fn):
    def wrapper(*args, **kwargs):
        startt = time.time()
        result = fn(*args, **kwargs)
        endt = time.time()
        print(f'Function {fn.__name__!r} executed in {endt - startt:.3f}s', file=sys.stderr)
        return result
    return wrapper

# Load models

@timed
def load_models():    
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    object_detector = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to("cuda")
    return object_detector, processor


def get_classnames_from_metadata(metadata):
    classnames = set()
    for obj in metadata["include"]:
        if "class" in obj:
            classnames.add(obj["class"])
    return list(classnames)


def get_largest_bbox(preds, idx, orig_size):
    """
    Extract largest contour bbox from CLIPSeg predictions, rescaled to original image size.
    preds[idx] is shape (1, Hmask, Wmask).
    orig_size: (width, height) of the original image
    """
    mask = torch.sigmoid(preds[idx][0]).cpu().numpy()
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)

    mask = (mask * 255).astype(np.uint8)

    # threshold --> binary mask
    _, bw_image = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)

    # find contours
    contours, _ = cv2.findContours(bw_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # choose contour with max area
    largest_cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_cnt)

    # rescale bbox to original image size
    mask_h, mask_w = mask.shape
    orig_w, orig_h = orig_size
    scale_x = orig_w / mask_w
    scale_y = orig_h / mask_h

    return [
        int(x * scale_x),
        int(y * scale_y),
        int((x + w) * scale_x),
        int((y + h) * scale_y),
    ]


def run_clipseg_object_detection(object_detector, processor, filepath, metadata):
    detected = defaultdict(list)
    classnames = get_classnames_from_metadata(metadata)
    prompts = [f"a {obj_name}" for obj_name in classnames]

    image = Image.open(filepath)
    orig_size = image.size  # (W, H)

    inputs = processor(text=prompts, images=[image] * len(prompts),
                       padding="max_length", return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = object_detector(**inputs)

    preds = outputs.logits.unsqueeze(1).cpu()  # [N, 1, Hmask, Wmask]

    for i in range(preds.shape[0]):
        bbox = get_largest_bbox(preds, i, orig_size)
        if bbox is not None:
            detected[classnames[i]].append([bbox])

    return detected


def relative_position(obj_a, obj_b):
    """Give position of A relative to B, factoring in object dimensions"""
    boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b
    #
    revised_offset = np.maximum(np.abs(offset) - POSITION_THRESHOLD * (dim_a + dim_b), 0) * np.sign(offset)
    if np.all(np.abs(revised_offset) < 1e-3):
        return set()
    #
    dx, dy = revised_offset / np.linalg.norm(offset)
    relations = set()
    
    if dx < -0.5: relations.add("to the left of")
    if dx > 0.5: relations.add("to the right of")
    if dy < -0.5: relations.add("above")
    if dy > 0.5: relations.add("below")
    
    return relations


def evaluate(args, objects, metadata):
    """
    Evaluate given image using detected objects on the global metadata specifications.
    Assumptions:
    * Metadata combines 'include' clauses with AND, and 'exclude' clauses with OR
    * All clauses are independent, i.e., duplicating a clause has no effect on the correctness
    * CHANGED: Color and position will only be evaluated on the most confidently predicted objects;
        therefore, objects are expected to appear in sorted order
    """
    
    correct = True
    reason = []
    matched_groups = []
    true_rels = []
    
    version = metadata.get('version', 1)
    
    # Check for expected objects
    for req in metadata.get('include', []):
        classname = req['class']
        matched = True
        found_objects = objects.get(classname, [])[:req['count']]
        if len(found_objects) < req['count']:
            correct = matched = False
            reason.append(f"expected {classname}>={req['count']}, found {len(found_objects)}")
        else:
            if 'position' in req and matched:
                # Relative position check
                expected_rel, target_group = req['position']
                
                ## to support previous version, where objects were placed differently
                if version == 1 and target_group > 0:
                    target_group *= 2
                
                if matched_groups[target_group] is None:
                    correct = matched = False
                    reason.append(f"no target for {classname} to be {expected_rel}")
                else:
                    for obj in found_objects:
                        for target_obj in matched_groups[target_group]:
                            if version == 1:
                                obj_a = target_obj
                                obj_b = obj
                            else:
                                obj_a = obj
                                obj_b = target_obj
                            
                            gt_rels = relative_position(obj_a, obj_b)
                            true_rels += list(gt_rels)
                            incorrect_spatial = False
                            if (expected_rel not in true_rels):
                                incorrect_spatial = True
                            
                            if incorrect_spatial:
                                correct = matched = False
                                reason.append(
                                    f"expected {classname} {expected_rel} target, found " +
                                    f"{' and '.join(true_rels)} target"
                                )
                                break
                        if not matched:
                            break
                        
        if matched:
            matched_groups.append(found_objects)
        else:
            matched_groups.append(None)
            
    # Check for non-expected objects
    for req in metadata.get('exclude', []):
        classname = req['class']
        if len(objects.get(classname, [])) >= req['count']:
            correct = False
            reason.append(f"expected {classname}<{req['count']}, found {len(objects[classname])}")
    
    return correct, "\n".join(reason), true_rels


def evaluate_image(args, filepath, metadata, object_detector, processor):
    detected_objects = run_clipseg_object_detection(object_detector, processor, filepath, metadata)
    is_correct, reason, true_rels = evaluate(args, detected_objects, metadata)
    
    gt_relation = []
    for true_rel in true_rels:
        if type(true_rel) is not dict:
            gt_relation = [true_rel]

    details_dict = {key: value[0] for key, value in detected_objects.items()}
    
    return {
        'filename': filepath,
        'tag': metadata['tag'],
        'prompt': metadata['prompt'],
        'correct': is_correct,
        'gt_relation': gt_relation,
        'true_rels': true_rels,
        'reason': reason,
        'metadata': json.dumps(metadata),
        'details': json.dumps(details_dict),
    }


def main(args, object_detector, processor):
    full_results = []
    for subfolder in os.listdir(args.imagedir):
        folderpath = os.path.join(args.imagedir, subfolder)
        if not os.path.isdir(folderpath) or not subfolder.isdigit():
            continue
        
        with open(os.path.join(folderpath, "metadata.jsonl")) as fp:
            metadata = json.load(fp)
        
        # Evaluate each image
        for imagename in os.listdir(os.path.join(folderpath, "samples")):
            imagepath = os.path.join(folderpath, "samples", imagename)
            if not os.path.isfile(imagepath) or not re.match(r"\d+\.png", imagename):
                continue
            
            result = evaluate_image(args, imagepath, metadata, object_detector, processor)
            result_path = Path(imagepath)
            result_path = result_path.with_name(result_path.stem + "__gen_eval_result.json")
            with open(result_path, "w") as result_f:
                json.dump(result, result_f, indent=4)
            
            full_results.append(result)
    
    # Save results
    if os.path.dirname(args.outfile):
        os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    with open(args.outfile, "w") as fp:
        pd.DataFrame(full_results).to_json(fp, orient="records", lines=True)
    
    # Dump metrics from the result output file  
    dump_metrics(args.outfile)


if __name__ == "__main__":
    args = parse_args()
    object_detector, processor = load_models()
    main(args, object_detector, processor)
    