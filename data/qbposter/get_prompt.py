import json
import random
import numpy as np
from matplotlib import pyplot as plt

import argparse

def process_json(json_path, template_path, train_image_root, val_image_root, d=4):
    # load template
    data = open(template_path, "r").read()
    # load json
    json_data = json.load(open(json_path, "r"))
    
    train_json, val_json = [], []
    for vid, poster in json_data.items():
        poster_id = vid # get id
        is_train = poster['split'] == 'train' # get train or val
        # image_name = poster['img'] # get image name
        image_name = vid + ".png" # get image name
        resolution = [poster['width'], poster['height']] # get resolution
        gt_content_raw = poster['boxes'] # get content
        
        gt_content = []
        # NOTICE: qbposter need a little bit of additional format processing
        for element in gt_content_raw:
            x_center, y_center, width, height = element['xc'], element['yc'], element['width'], element['height']
            left, top, right, bottom = x_center - width // 2, y_center - height // 2, x_center + width // 2, y_center + height // 2
            gt_content.append({
                'label': element['label'],
                'box': [
                    round(left / resolution[0], d),
                    round(top / resolution[1], d),
                    round(right / resolution[0], d),
                    round(bottom / resolution[1], d)
                ]
            })
        
        # get prompt
        content_for_prompt = []
        for element in gt_content:
            element_for_prompt = element.copy()
            element_for_prompt['box'] = [] # mask the ground truth bounding box
            content_for_prompt.append(element_for_prompt)
        prompt = data.format(
            N=len(gt_content),
            resolution=resolution,
            domain_name="social media promotion poster with qbposter style",
            json_data=json.dumps(content_for_prompt)
        )

        out_data = {}
        out_data["id"] = poster_id
        out_data["image"] = (train_image_root if is_train else val_image_root) + image_name
        
        out_data["conversations"] = [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": f"Sure! Here is the design results: {json.dumps(gt_content)}"},
        ]

        if is_train:
            train_json.append(out_data)
        else:
            val_json.append(out_data)
        
    return train_json, val_json

def main(args):
    # process json
    train_data, val_data = process_json(args.json_path, args.template_path, args.train_image_root, args.val_image_root, args.d)

    # write in a json format
    output_train_path = args.output_dir + "qbposter_train_instruct.json"
    json.dump(train_data, open(output_train_path, 'w', encoding='utf-8'), indent=2)
    print(f"Train data saved in {output_train_path}")
    
    output_val_path = args.output_dir + "qbposter_val_instruct.json"
    json.dump(val_data, open(output_val_path, 'w', encoding='utf-8'), indent=2)
    print(f"Val data saved in {output_val_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-path", type=str, default="data/prompt_template.txt")
    parser.add_argument("--json-path", type=str, default="data/qbposter/raw/annotations.json")
    parser.add_argument("--train-image-root", type=str, default="qbposter/raw/inpainted_1d5x/")
    parser.add_argument("--val-image-root", type=str, default="qbposter/raw/inpainted_1x/")
    parser.add_argument("--output-dir", type=str, default="data/qbposter/")
    parser.add_argument("-d", type=int, default=4, help='round to D decimal places')
    args = parser.parse_args()
    main(args)