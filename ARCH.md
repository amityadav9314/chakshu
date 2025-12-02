# Chakshu (चक्षु) - Architecture & Context

> **Note to AI Agents:** This file serves as the source of truth for the project's context, architecture, and coding standards. Read this first to understand the "What", "Why", and "How" of Chakshu.

## 1. Project Identity
- **Name:** Chakshu (Sanskrit for "Eye")
- **Goal:** Develop an AI/ML model to accurately identify and track humans in CCTV video footage.
- **Domain:** Computer Vision, Surveillance, Object Detection.

## 2. Technology Stack
- **Language:** Python 3.12+
- **Dependency Manager:** `uv` (Fast, reliable)
- **Core Libraries:**
  - **OpenCV (`opencv-python`):** Image processing and video handling.
  - **NumPy:** Numerical operations.
  - **Matplotlib:** Data visualization and image display.
  - **Rich:** Beautiful terminal output with progress bars and formatting.
  - **YOLOv8 (`ultralytics`):** Object detection model.
  - **PyTorch:** (Planned/In-progress) Deep learning framework for model training.
- **Data Format:** COCO (Common Objects in Context) for annotations.
- **Hardware Acceleration:**
  - **AMD GPU:** Supported via ROCm (Linux only).
  - **Requirement:** For RDNA 3 cards (e.g., RX 7800 XT), set `HSA_OVERRIDE_GFX_VERSION=11.0.0`.

## 3. Data & Environment
- **Dataset:** COCO format.
- **OS Support:** Native Ubuntu (Linux).
- **Locations:**
  - **Data:** Configured in `src/constants.py`.
- **Configuration:** 
  - Paths managed via `src/constants.py`.

## 4. Coding Guidelines & Rules
- **Virtual Environment:** ALWAYS use `.venv` for all development work. `uv` automatically manages this.
- **Dependency Management:** ALWAYS use `uv` for adding/removing packages (`uv add <package>`, `uv run <script>`).
- **Code Style:**
  - Follow PEP 8.
  - Use type hints where helpful.
  - Keep functions small and focused (Single Responsibility Principle).
- **Project Structure:**
  - `src/`: Core logic, data processing, model definitions.
  - `tests/`: Unit and integration tests.
  - `pyproject.toml`: Configuration source of truth.
- **Documentation:** Update `README.md` and `ARCH.md` as the project evolves.

## 5. COCO Annotation Format Reference

Understanding the COCO format is critical for working with our dataset. Each annotation is a JSON object with the following structure:

### Annotation Structure
```json
{
  "area": 3452.78,                    // Area of bounding box in pixels²
  "bbox": [297.27, 107.87, 58.73, 108.06],  // [x, y, width, height]
  "category_id": 1,                   // Object category (1 = person in COCO)
  "id": 559505,                       // Unique annotation ID
  "image_id": 440830,                 // Which image this belongs to
  "iscrowd": 0,                       // 0 = single object, 1 = crowd
  "segmentation": [[x1,y1, x2,y2, ...]]  // Polygon points for precise outline
}
```

### Key Fields
- **`bbox`**: Bounding box in `[x, y, width, height]` format. `(x, y)` is the top-left corner.
- **`category_id`**: Object class. In COCO: `1` = person, `3` = car, `8` = truck, etc.
- **`segmentation`**: Polygon coordinates `[x1, y1, x2, y2, ...]` for pixel-perfect object outline (more precise than bbox).
- **`area`**: Total pixel area. Useful for filtering small/large objects.
- **`iscrowd`**: `0` for individual objects, `1` for groups too dense to annotate individually.
- **`image_id`**: Links annotation to specific image. Multiple annotations can share the same `image_id`.

### Usage in Chakshu
- We primarily use **`bbox`** for drawing rectangles in visualization.
- **`segmentation`** is available for future precise masking/segmentation tasks.
- Filter by **`category_id`** to focus on persons (1), vehicles (2, 3, 4, 6, 8), etc.

## 6. Current State
- **Phase:** Initial Setup & Data Inspection.
- **Status:**
  - Project initialized with `uv`.
  - Data inspection script (`src/inspection/01_data_inspection.py`) is functional and has verified the dataset.
  - Visual quality check script (`src/inspection/02_visual_quality_check.py`) is ready to visualize annotated samples.
  - False triggers detection script (`src/inspection/03_check_for_false_triggers.py`) with parallel processing (24 CPU cores) for quality checks.
  - Unwanted classes (animals) identified in the dataset.
