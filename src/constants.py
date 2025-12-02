"""
Chakshu Constants

Configuration for the Chakshu project (Native Ubuntu).
"""

import os
from pathlib import Path

# TODO: Update these paths after mounting
# Defaulting to a likely mount point or placeholder
DATA_ROOT = Path("/media/aky/Data/AIML/Data") 
IMG = DATA_ROOT / "COCO_TD/train2017/train2017"
ANNOTATIONS_PATH = DATA_ROOT / "COCO_TD/annotations_trainval2017/annotations/instances_train2017.json"

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
ARCHIVE_DIR = PROJECT_ROOT / "src" / "archive"
INSPECTION_DIR = PROJECT_ROOT / "src" / "inspection"

__all__ = ['IMG', 'ANNOTATIONS_PATH', 'PROJECT_ROOT']
