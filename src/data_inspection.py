import json
import cv2
import os
from collections import Counter
from constants import IMG, ANNOTATIONS_PATH


# For COCO format datasets
def inspect_coco_dataset(json_path, image_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Check class distribution
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    annotations = data['annotations']

    # Count instances per class
    class_counts = Counter([ann['category_id'] for ann in annotations])

    print("=== Class Distribution ===")
    for cat_id, count in class_counts.items():
        print(f"{categories[cat_id]}: {count} instances")

    # Check for unwanted classes (animals you want to exclude)
    unwanted = ['dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
    print("\n=== Unwanted Classes Present ===")
    for cat_id, name in categories.items():
        if name.lower() in unwanted:
            count = class_counts.get(cat_id, 0)
            print(f"[WARNING] {name}: {count} instances (FILTER THIS)")

    return data, categories


# Run inspection
if __name__ == '__main__':
    data, categories = inspect_coco_dataset(ANNOTATIONS_PATH, IMG)
