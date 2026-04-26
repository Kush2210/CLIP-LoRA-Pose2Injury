# Keypoint-Guided Gaussian Segmentation + CLIP-LoRA Pipeline

This project detects human pose keypoints, builds limb-focused Gaussian masks, and classifies limb condition using CLIP + LoRA.

## Preview

![Batch report preview](Results/batch_reports/001_report.jpg)

## What This Repo Contains

- `pose_gaussian_only.py`: pose + limb Gaussian masking for one image.
- `run_pose_then_clip.py`: end-to-end single-image pipeline (masking -> CLIP/LoRA classification).
- `batch_testing_report.py`: batch report generator over `DATASET/images` with visualization grids and CSV outputs.
- `analyze_batch_results.py`: computes limb-level accuracy, precision, recall, F1, probability stats, and confusion matrix files.
- `cv_project_usage.ipynb`: the full walkthrough notebook with final image display and limb context.
- `notebooks/01_base_clip_limb_status.ipynb`: base CLIP on the raw image.
- `notebooks/02_yolo_pose_limb_crops_clip.ipynb`: YOLO pose + Gaussian masking with base CLIP.
- `notebooks/03_yolo_gaussian_lora_clip.ipynb`: YOLO pose + Gaussian masking with CLIP + LoRA and the compact limb table.
- `CLIP-LoRA/`: CLIP-LoRA training/inference code and dataset adapters.
- `DATASET/`: current 12-image dataset and `ground_truth.csv`.
- `testing/`: sample images used by the notebook walkthroughs.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Model Files

- Pose model default: `yolov8n-pose.pt`
- Base CLIP weights are pulled from the web on first run and cached in `~/.cache/clip/`.
- LoRA checkpoint example: `CLIP-LoRA/weights/lora_weights_960_2.9566854533582632e-05.pt`

Notes:

- If `yolov8n-pose.pt` is not present locally, Ultralytics may auto-download it on first run.
- The main notebook and batch flows currently use CLIP + LoRA directly.

## Class Labels and Prompts

Default class order:

1. `injury`
2. `no_injury`

The project uses prompt-based CLIP text features with prompt ensembling for better stability.

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

### 3) Batch report over `DATASET/images`

```bash
python batch_testing_report.py
```

Outputs are written to:

- `Results/batch_reports/*_report.jpg`
- `Results/batch_reports/confusion_matrix.csv`
- `Results/batch_reports/confusion_matrix.png`
- `Results/batch_reports/summary.csv`
- `Results/batch_reports/part_probabilities.csv`
- `Results/batch_reports/ground_truth_vs_predicted.csv`
- `Results/batch_reports/part_masks/*_gaussian_mask.png`

Run the metrics pass with:

```bash
python analyze_batch_results.py
```

That prints limb-level accuracy, precision, recall, F1, probability stats, and writes the confusion matrix files.


Outputs are written to:

- `Results/report_assets_positive/metrics_summary.json`
- `Results/report_assets_positive/metrics_table.csv`
- `Results/report_assets_positive/confusion_matrix.png`
- `Results/report_assets_positive/roc_ovr.png`
- `Results/report_assets_positive/class_distribution.png`
- `Results/report_assets_positive/metrics_bar.png`
- `Results/report_assets_positive/sample_group_panel.png`
- `Results/report_assets_positive/sample_predictions.png`
- `PRESENTATION_EVIDENCE_CHECKLIST.md`

## Notebook Comparison

Run `cv_project_usage.ipynb` for the full walkthrough:

- pose detection
- keypoint/skeleton visualization
- Gaussian limb masks
- per-limb CLIP + LoRA predictions
- final image display with `injury` / `no_injury` title
- limb-by-limb context table at the end

The focused notebook variants in `notebooks/` are useful for side-by-side comparison:

- `01_base_clip_limb_status.ipynb` is the baseline: raw image, base CLIP only.
- `02_yolo_pose_limb_crops_clip.ipynb` adds YOLO pose and Gaussian limb masking, still with base CLIP.
- `03_yolo_gaussian_lora_clip.ipynb` is the CLIP + LoRA version of the same limb pipeline and uses the compact five-column table:
  `Limb`, `Predicted Class`, `Binary Label`, `Injury Score`, `No-Injury Prob`.

If you want to compare model behavior, the clearest progression is `01 -> 02 -> 03`.
=