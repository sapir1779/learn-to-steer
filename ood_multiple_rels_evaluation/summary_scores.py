# Get results of evaluation

import argparse
import os
import json
import numpy as np
import pandas as pd


def dump_metrics(filename):
    # Load results
    df = pd.read_json(filename, orient="records", lines=True)

    # Measure overall success
    total_images = len(df)
    total_prompts = len(df.groupby('metadata'))
    correct_imgs_percentage = df['correct'].mean()
    correct_prompts_percentage = df.groupby('metadata')['correct'].any().mean()
    print("Summary")
    print("=======")
    print(f"Total images: {total_images}")
    print(f"Total prompts: {total_prompts}")
    print(f"% correct images: {correct_imgs_percentage:.2%}")
    print(f"% correct prompts: {correct_prompts_percentage:.2%}")
    print()

    # By group
    task_scores_dict = {}
    task_scores = []
    print("Task breakdown")
    print("==============")
    for tag, task_df in df.groupby('tag', sort=False):
        tag_score = task_df['correct'].mean()
        total_correct = task_df['correct'].sum()
        total_images = len(task_df)
        task_scores_dict[tag] = {
            "tag_score": tag_score,
            "total_correct": int(total_correct),
            "total_images": int(total_images),
        }
        task_scores.append(tag_score)
        print(f"{tag:<16} = {tag_score:.2%} ({total_correct} / {total_images})")

    print()

    overall_score = np.mean(task_scores)
    print(f"Overall score (avg. over tasks): {overall_score:.5f}")

    ## Dump metrics to json as well
    metrics_path = os.path.join(os.path.dirname(filename), "metrics.json")
    # metrics_path = os.path.join(os.path.dirname(filename), "metrics__new.json")
    metrics = {
        "correct_imgs_percentage": correct_imgs_percentage,
        "correct_prompts_percentage": correct_prompts_percentage,
        "total_images": int(total_images),
        "total_prompts": int(total_prompts),
        "task_scores_dict": task_scores_dict,
        "overall_score": overall_score,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    args = parser.parse_args()
    dump_metrics(args.filename)