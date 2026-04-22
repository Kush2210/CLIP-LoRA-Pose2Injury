# Keypoint-Guided Gaussian Segmentation + CLIP-LoRA Pipeline

This project detects human pose keypoints, builds limb-focused Gaussian masks, and classifies limb condition using CLIP with optional LoRA weights.

## What This Repo Contains

- `pose_gaussian_only.py`: pose + limb Gaussian masking for one image.
- `run_pose_then_clip.py`: end-to-end single-image pipeline (masking -> CLIP/LoRA classification).
- `batch_testing_report.py`: batch report generator with visualization grids and CSV outputs.
- `cv_project_usage.ipynb`: step-by-step notebook demo for the same pipeline.
- `CLIP-LoRA/`: CLIP-LoRA training/inference code and dataset adapters.
- `testing/`: sample test images.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Model Files

- Pose model default: `yolov8n-pose.pt`
- LoRA checkpoint example: `CLIP-LoRA/weights/lora_weights_960_2.9566854533582632e-05.pt`

Notes:

- If `yolov8n-pose.pt` is not present locally, Ultralytics may auto-download it on first run.
- LoRA is optional. Without `--use_lora`, inference runs in zero-shot CLIP mode.

## Class Labels and Prompts

Default class order:

1. `injury`
2. `no_injury`
3. `injury_and_amputation`

The project uses prompt-based CLIP text features. The notebook and batch script currently use prompt ensembling for better stability.

## Quickstart

### 1) Pose + Gaussian mask only

```bash
python pose_gaussian_only.py \
  --input testing/001.png \
  --output Results/splatted_output.jpg \
  --limb left_leg
```

### 2) End-to-end single-image inference

```bash
python run_pose_then_clip.py \
  --input testing/001.png \
  --mode predict \
  --use_lora \
  --lora_save_path CLIP-LoRA/weights/lora_weights_960_2.9566854533582632e-05.pt
```

Important:

- `run_pose_then_clip.py` now uses the current Python interpreter (`sys.executable`) by default.
- You can override interpreter with `--python_executable /path/to/python` if needed.

### 3) Batch report over `testing/`

```bash
python batch_testing_report.py
```

Outputs are written to:

- `Results/batch_reports/*.jpg`
- `Results/batch_reports/summary.csv`
- `Results/batch_reports/part_probabilities.csv`

## Notebook

Run `cv_project_usage.ipynb` for an interactive walkthrough:

- pose detection
- keypoint/skeleton visualization
- limb Gaussian masks
- CLIP + LoRA probability tables
- image-level injury summary

## Submission Notes

This repo includes `.gitignore` for generated artifacts (`Results/`, `__pycache__/`, notebook checkpoints, etc.) so reruns do not pollute source control.

For final submission, keep source files and sample inputs; generated outputs are optional unless explicitly requested by your course.

## Troubleshooting

### LoRA checkpoint error

- Confirm the file path passed via `--lora_save_path` (or notebook variable) exists.

### Torch attention compatibility

- `CLIP-LoRA/loralib/layers.py` includes a fallback attention path for environments that do not expose `scaled_dot_product_attention`.

### No person detected

- Try a clearer image, a different crop, or a different limb.

## Minimal Repo Review Checklist

Before submitting:

1. `pip install -r requirements.txt` succeeds.
2. `python run_pose_then_clip.py --input testing/001.png --mode predict` runs.
3. `python batch_testing_report.py` runs and writes reports.
4. README commands match actual file paths in your repo.
