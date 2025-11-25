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
  - **YOLOv8 (`ultralytics`):** Object detection model.
  - **PyTorch:** (Planned/In-progress) Deep learning framework for model training.
- **Data Format:** COCO (Common Objects in Context) for annotations.

## 3. Data & Environment
- **Dataset:** COCO format.
- **Locations (Windows):**
  - Images: `F:\Soft\AIML\Data\COCO_TD\train2017`
  - Annotations: `F:\Soft\AIML\Data\COCO_TD\annotations_trainval2017\annotations\instances_train2017.json`
- **Configuration:** Paths are managed in `src/constants.py`.

## 4. Coding Guidelines & Rules
- **Dependency Management:** ALWAYS use `uv` for adding/removing packages (`uv add <package>`, `uv run <script>`).
- **Code Style:**
  - Follow PEP 8.
  - Use type hints where helpful.
  - Keep functions small and focused (Single Responsibility Principle).
- **Project Structure:**
  - `src/`: Core logic, data processing, model definitions.
  - `tests/`: Unit and integration tests.
  - `pyproject.toml`: Configuration source of truth.
- **Documentation:** Update `README.md` and `arch.md` as the project evolves.

## 5. Current State
- **Phase:** Initial Setup & Data Inspection.
- **Status:**
  - Project initialized with `uv`.
  - Data inspection script (`src/data_inspection.py`) is functional and has verified the dataset.
  - Unwanted classes (animals) identified in the dataset.
