"""
COCO Dataset Visual Quality Check Script

This script provides visual verification of COCO dataset annotations by displaying
random sample images with all their bounding boxes and labels overlaid.

Purpose:
    - Visually verify annotation quality and accuracy
    - Spot-check that bounding boxes correctly align with objects
    - Identify mislabeled or incorrectly annotated images
    - Understand what the training data looks like before model training
    - Ensure data quality before investing time in model training

Key Features:
    - Randomly samples images (not individual annotations)
    - Displays ALL annotations per image (not just one)
    - Filters for surveillance-relevant categories: person, car, truck, bus, motorcycle
    - Shows annotation count in title for quick verification

Usage:
    Run directly from command line:
        $ uv run src/inspection/02_visual_quality_check.py
    
    Or import and use programmatically:
        from src.inspection.visual_quality_check import visualize_random_samples
        visualize_random_samples(data, categories, image_dir, num_samples=5)

Output:
    - Opens matplotlib windows showing images with:
        - Green bounding boxes around detected objects
        - Category labels above each box
        - Image filename and annotation count in title

Important Notes:
    - This script samples IMAGES, not individual annotations
    - If an image has 10 people, ALL 10 will be shown with bounding boxes
    - Only shows images that contain at least one target category object

Author: Chakshu Project
"""

import os
import json
import random

import cv2
import matplotlib.pyplot as plt
from src.constants import IMG, ANNOTATIONS_PATH


def visualize_random_samples(data, categories, image_dir, num_samples=5):
    """
    Visualize random images with ALL their annotations overlaid.
    
    This function randomly selects images from the dataset that contain surveillance-
    relevant objects (person, vehicles) and displays them with all bounding boxes
    and category labels drawn on the image.
    
    Args:
        data (dict): COCO dataset structure containing:
            - 'images': List of image metadata
            - 'annotations': List of annotation objects
            - 'categories': List of category definitions
        categories (dict): Mapping of category_id -> category_name
        image_dir (str): Directory path containing the actual image files
        num_samples (int, optional): Number of random images to display. Default: 5
    
    Returns:
        None: Displays images using matplotlib.pyplot.show()
    
    Behavior:
        1. Filters for target categories: person, car, truck, bus, motorcycle
        2. Finds all images containing at least one target object
        3. Randomly samples num_samples images from this set
        4. For each image:
            - Loads the image from disk
            - Finds ALL annotations for that image
            - Draws green bounding boxes (2px thick) for each annotation
            - Adds category label text above each box
            - Displays using matplotlib with annotation count in title
    
    Visual Output:
        - Bounding boxes: Green rectangles (RGB: 0, 255, 0)
        - Labels: Green text above boxes using HERSHEY_SIMPLEX font
        - Title format: "Image: {filename} | Annotations: {count}"
        - Figure size: 12x10 inches for better visibility
    
    Error Handling:
        - Skips images that cannot be loaded from disk
        - Prints warning message for failed image loads
    
    Example:
        >>> # Load COCO data
        >>> with open('annotations.json') as f:
        ...     data = json.load(f)
        >>> categories = {cat['id']: cat['name'] for cat in data['categories']}
        >>> 
        >>> # Visualize 10 random samples
        >>> visualize_random_samples(data, categories, 'images/', num_samples=10)
        # Opens 10 matplotlib windows showing annotated images
    
    Notes:
        - This function samples IMAGES, not individual annotations
        - All objects in each image are shown, not just one
        - Only images with target categories are considered
        - Images are displayed one at a time (blocking)
    """
    # Define surveillance-relevant categories
    target_categories = ['person', 'car', 'truck', 'bus', 'motorcycle']
    target_cat_ids = [cat['id'] for cat in data['categories']
                      if cat['name'] in target_categories]

    # Find all images that have at least one target annotation
    target_image_ids = set()
    for ann in data['annotations']:
        if ann['category_id'] in target_cat_ids:
            target_image_ids.add(ann['image_id'])
    
    # Convert to list and randomly sample images
    target_image_ids = list(target_image_ids)
    sample_image_ids = random.sample(target_image_ids, min(num_samples, len(target_image_ids)))

    # Process each sampled image
    for img_id in sample_image_ids:
        # Get image metadata
        img_info = next(img for img in data['images'] if img['id'] == img_id)
        img_path = os.path.join(image_dir, img_info['file_name'])

        # Load image from disk
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            continue
        # Convert BGR (OpenCV) to RGB (matplotlib)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get ALL annotations for this image (filtered by target categories)
        img_annotations = [ann for ann in data['annotations'] 
                          if ann['image_id'] == img_id and ann['category_id'] in target_cat_ids]

        # Draw all bounding boxes and labels
        for ann in img_annotations:
            # Extract bounding box coordinates
            # COCO format: [x, y, width, height] where (x,y) is top-left corner
            x, y, w, h = map(int, ann['bbox'])
            
            # Draw green rectangle (2px thick)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add category label above the box
            cat_name = categories[ann['category_id']]
            cv2.putText(img, cat_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

        # Display the annotated image
        plt.figure(figsize=(12, 10))
        plt.imshow(img)
        plt.title(f"Image: {img_info['file_name']} | Annotations: {len(img_annotations)}")
        plt.axis('off')
        plt.show()


# Script entry point
if __name__ == '__main__':
    # Load COCO annotations from constants
    with open(ANNOTATIONS_PATH, 'r') as f:
        data = json.load(f)
    
    # Build category mapping: category_id -> category_name
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Visualize 10 random samples
    visualize_random_samples(data, categories, IMG, num_samples=10)
