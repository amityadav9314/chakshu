import json
import os
from pathlib import Path
from rich.console import Console
from rich.progress import track
from src.constants import ANNOTATIONS_PATH, DATA_ROOT

console = Console()

def convert_coco_to_yolo():
    """
    Convert COCO annotations to YOLO format for 'person' class.
    """
    console.print(f"[bold blue]Loading annotations from {ANNOTATIONS_PATH}...[/bold blue]")
    
    with open(ANNOTATIONS_PATH, 'r') as f:
        data = json.load(f)
    
    images = {img['id']: img for img in data['images']}
    annotations = data['annotations']
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Find person category ID
    person_id = None
    for cat_id, name in categories.items():
        if name == 'person':
            person_id = cat_id
            break
            
    if person_id is None:
        console.print("[bold red]Error: 'person' category not found![/bold red]")
        return

    console.print(f"[green]Found 'person' category ID: {person_id}[/green]")
    
    # Create labels directory
    labels_dir = DATA_ROOT / "COCO_TD/train2017/labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[blue]Labels will be saved to: {labels_dir}[/blue]")

    # Process annotations
    count = 0
    
    # Group annotations by image
    img_anns = {}
    for ann in annotations:
        if ann['category_id'] != person_id:
            continue
            
        img_id = ann['image_id']
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)

    console.print(f"[blue]Processing {len(img_anns)} images containing persons...[/blue]")

    for img_id, anns in track(img_anns.items(), description="Converting..."):
        img_info = images.get(img_id)
        if not img_info:
            continue
            
        img_w = img_info['width']
        img_h = img_info['height']
        file_name = Path(img_info['file_name']).stem
        
        label_file = labels_dir / f"{file_name}.txt"
        
        with open(label_file, 'w') as f:
            for ann in anns:
                bbox = ann['bbox'] # x, y, w, h
                
                # Normalize to center_x, center_y, w, h
                x_center = (bbox[0] + bbox[2] / 2) / img_w
                y_center = (bbox[1] + bbox[3] / 2) / img_h
                width = bbox[2] / img_w
                height = bbox[3] / img_h
                
                # YOLO format: class_id x_center y_center width height
                # We map person_id to class 0
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        count += 1

    console.print(f"[bold green]Successfully converted {count} labels![/bold green]")

if __name__ == "__main__":
    convert_coco_to_yolo()
