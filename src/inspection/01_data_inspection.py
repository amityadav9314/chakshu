"""
COCO Dataset Inspection Script

This script analyzes the COCO format dataset to provide statistical insights about
class distribution and identify unwanted object categories.

Purpose:
    - Verify dataset integrity and structure
    - Understand class distribution across the dataset
    - Identify unwanted classes (e.g., animals) that should be filtered out
    - Provide quick statistics before training

Usage:
    Run directly from command line:
        $ uv run src/inspection/01_data_inspection.py
    
    Or import and use programmatically:
        from src.inspection.data_inspection import inspect_coco_dataset
        data, categories = inspect_coco_dataset(json_path, image_dir)

Output:
    - Prints class distribution (all 80 COCO classes with instance counts)
    - Highlights unwanted classes that should be filtered for surveillance use case
    - Returns loaded data and category mappings for further processing

Author: Chakshu Project
"""

import json
from collections import Counter

from src.constants import IMG, ANNOTATIONS_PATH


def inspect_coco_dataset(json_path, image_dir):
    """
    Inspect COCO format dataset and analyze class distribution.
    
    This function loads a COCO annotations file and provides statistical analysis
    of the dataset, including class distribution and identification of unwanted
    categories that are not relevant for CCTV surveillance (e.g., animals).
    
    Args:
        json_path (str): Path to COCO format JSON annotation file
        image_dir (str): Directory containing the images (not used in current implementation,
                        but kept for API consistency)
    
    Returns:
        tuple: (data, categories) where:
            - data (dict): Complete COCO dataset structure with keys:
                - 'images': List of image metadata
                - 'annotations': List of annotation objects
                - 'categories': List of category definitions
            - categories (dict): Mapping of category_id -> category_name
    
    Prints:
        - Complete class distribution showing all categories and their instance counts
        - Warning list of unwanted classes (animals) that should be filtered
    
    Example:
        >>> data, cats = inspect_coco_dataset('annotations.json', 'images/')
        === Class Distribution ===
        person: 262465 instances
        car: 43867 instances
        ...
        === Unwanted Classes Present ===
        [WARNING] dog: 5508 instances (FILTER THIS)
        [WARNING] cat: 4768 instances (FILTER THIS)
    """
    # Load COCO annotations
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Build category mapping: category_id -> category_name
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    annotations = data['annotations']

    # Count instances per class using Counter for efficiency
    class_counts = Counter([ann['category_id'] for ann in annotations])

    # Display complete class distribution
    print("=== Class Distribution ===")
    for cat_id, count in class_counts.items():
        print(f"{categories[cat_id]}: {count} instances")

    # Identify unwanted classes for surveillance use case
    # These are primarily animals that are not relevant for CCTV human detection
    unwanted = ['dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
    
    print("\n=== Unwanted Classes Present ===")
    for cat_id, name in categories.items():
        if name.lower() in unwanted:
            count = class_counts.get(cat_id, 0)
            print(f"[WARNING] {name}: {count} instances (FILTER THIS)")

    return data, categories


# Script entry point
if __name__ == '__main__':
    # Run inspection using paths from constants
    data, categories = inspect_coco_dataset(ANNOTATIONS_PATH, IMG)
