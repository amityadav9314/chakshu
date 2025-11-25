# Chakshu (चक्षु)

**Chakshu** (Sanskrit for "Eye") is an AI/ML project dedicated to identifying and tracking humans in CCTV video footage.

## Overview
The primary goal of this project is to develop a robust machine learning model capable of processing surveillance video to detect human presence with high accuracy. This tool aims to assist in security monitoring and automated surveillance analysis.

## Prerequisites
- Python 3.12+
- `uv` package manager

## Setup
This project uses [uv](https://github.com/astral-sh/uv) for fast dependency management.

1. **Install dependencies:**
   ```bash
   uv sync
   ```

## Usage
### Data Inspection
To inspect the training data (COCO format):
```bash
uv run src/01_data_inspection.py
```

## Project Structure
- [ARCH.md](ARCH.md): Project architecture, context, and rules.
- `src/`: Source code for data processing and model definitions.
- `pyproject.toml`: Project configuration and dependencies.
