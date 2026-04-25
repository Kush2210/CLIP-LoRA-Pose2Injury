# Detailed Project Report
## Keypoint-Guided Gaussian Segmentation with CLIP-LoRA for Injury Assessment

## 1. Abstract
This report presents an end-to-end computer vision pipeline for injury-focused visual analysis using keypoint-guided Gaussian segmentation and pretrained CLIP/CLIP-LoRA classification. The method first estimates human keypoints with a YOLOv8 pose model, then creates a limb-focused Gaussian splatting map to segment or emphasize likely injury regions, and finally predicts injury classes using a pretrained CLIP ViT-B/16 model (with optional pretrained LoRA adapters at inference time). The workflow is inference-oriented, lightweight, interpretable, and suitable for rapid triage-style decision support.

## 2. Problem Statement and Objective
In emergency-response and defense workflows, operators must quickly localize suspected injury regions and infer severity from noisy visual streams. Manual analysis is slow and inconsistent under time pressure.

Project objective:
- Localize informative body regions using keypoints.
- Generate a focused segmented output through Gaussian splatting.
- Classify injury category using pretrained CLIP with optional pretrained LoRA adaptation.
- Produce explainable artifacts (plots, confusion matrix, sample predictions) for reporting.

Important clarification:
- No custom dataset was used to train a new classifier in this implementation.
- The classification stage relies on pretrained model capabilities and prompt-based inference.

## 3. Full Methodology

### 3.1 Pipeline Overview
The implemented pipeline has three major stages:
1. Pose/keypoint extraction using YOLOv8-pose.
2. Keypoint-driven Gaussian segmentation (splatting + masking).
3. CLIP-LoRA inference for wound severity classification.

An orchestration script runs this flow in sequence:
- Pose + Gaussian stage: pose_gaussian_only.py
- CLIP-LoRA stage: CLIP-LoRA/infer_single_wound.py
- End-to-end runner: run_pose_then_clip.py

### 3.2 Stage A: Pose and Keypoint Extraction
Input image is passed to YOLOv8 pose model, which predicts body joints. The current implementation supports limb-specific keypoint triplets:
- right_arm: [6, 8, 10]
- left_arm: [5, 7, 9]
- right_leg: [12, 14, 16]
- left_leg: [11, 13, 15]

Default configuration uses right_leg.

### 3.3 Stage B: Keypoint-Guided Gaussian Segmentation
For each adjacent pair of selected keypoints, spread scale is estimated from Euclidean distance:

$$
\sigma = \max(k\cdot ||p_i - p_{i+1}||,\, 1.0), \quad k=0.15
$$

A normalized 2D Gaussian kernel is generated around each point and midpoint:

$$
G(x,y) = \frac{1}{Z}\exp\left(-\frac{x^2+y^2}{2\sigma^2}\right)
$$

where $Z$ normalizes the kernel. The weighted accumulation over original pixel intensities forms a splatted intensity map. A binary range mask then keeps only activated regions:
- mask = inRange(splatted, [1,1,1], [255,255,255])
- segmented output = bitwise_and(original, original, mask)

This stage behaves like soft geometric segmentation, emphasizing limb-centered candidate injury zones while suppressing background clutter.

### 3.4 Stage C: CLIP-LoRA Classification
The segmented output is passed to CLIP:
- Backbone: ViT-B/16
- Image encoder output normalized
- Text prompts generated from class names and template
- Cosine similarity logits scaled by 100 and softmaxed

LoRA adaptation details from inference/training config:
- LoRA inserted into attention projections q, k, v
- Rank r = 2
- Alpha = 1
- Dropout = 0.25
- Encoder mode supports text, vision, or both (default both)

Operational mode used in this report:
- Inference-only image classification with pretrained weights.
- No additional dataset training performed in the reported run.
- Prediction is produced as class probabilities and top-k candidates.

Practical interpretation:
- The keypoint + Gaussian step acts as a region-prior module.
- The CLIP text-image similarity step acts as a semantic decision module.
- Together, they form a weakly supervised or label-light pipeline for scenario analysis.

### 3.5 Why This Method Works
- Keypoints provide explicit body geometry priors.
- Gaussian splatting provides smooth spatial weighting instead of hard cropping.
- CLIP adds strong language-aligned semantics for low-data settings.
- LoRA adapts CLIP efficiently with a small trainable parameter budget.

## 4. Dataset Details

### 4.1 Dataset Usage Statement
No custom labeled dataset was used to train the classification model for this report.

The classifier is used in pretrained inference mode:
- Pretrained CLIP ViT-B/16 provides image-text alignment.
- Optional LoRA checkpoints can be loaded as pretrained adaptation weights.
- Inference is performed on input images after keypoint-guided Gaussian segmentation.

### 4.2 Input Image Sources for Inference
Although no training dataset was used, input images are required for pipeline execution. In this project, inference images are sourced from workspace image folders and runtime inputs, for example:
- CLIP-LoRA/DATASET/test/frames (as an input image pool)
- User-provided single images through command-line arguments

### 4.3 Effective Data Attributes (Inference Perspective)
- RGB image tensor
- Pose keypoints (up to 17 keypoints per detected person)
- Limb-selected keypoint subset (3 points for arm/leg path)
- Gaussian splatted intensity mask
- Final segmented image passed to classifier
- Class probability vector from CLIP/CLIP-LoRA

### 4.4 Preprocessing and Representation
- Input image decoding and RGB conversion
- Pose keypoint extraction via YOLOv8-pose
- Sigma computation from inter-keypoint distances
- Gaussian kernel aggregation around joints and midpoints
- Masked image generation for region-focused classification
- CLIP preprocessing (resize and normalization)

## 5. Experimental Setup
- OS: Linux
- Main stack: Python, PyTorch, OpenCV, Ultralytics YOLOv8
- CLIP implementation under CLIP-LoRA/clip
- End-to-end command wrapper: run_pose_then_clip.py

Inference flow used for reporting:
1. Run pose + Gaussian segmentation.
2. Run CLIP/CLIP-LoRA single-image prediction.
3. Collect visual artifacts and output JSON for reporting.
4. Notebook evidence is also recorded in `notebooks/01_base_clip_limb_status.ipynb`, `notebooks/02_yolo_pose_limb_crops_clip.ipynb`, and `notebooks/03_yolo_gaussian_lora_clip.ipynb`.

Training status:
- No new classifier training was performed in this report.
- No train/validation split experiment was executed for benchmark claims.

## 6. Results

### 6.1 Result Type Clarification
Because no labeled benchmark dataset was used for model training/evaluation in this report run, strict supervised metrics such as accuracy, precision, recall, F1, ROC-AUC, and confusion matrix are not treated as primary validated claims.

Instead, the results are reported as:
- Qualitative segmentation quality
- Class probability outputs from pretrained CLIP/CLIP-LoRA
- Visual evidence from generated artifacts

### 6.2 Qualitative Results Summary

| Component | Observation |
|---|---|
| Pose keypoint detection | Correctly localizes body joints for selected limb in clear frames |
| Gaussian splatting map | Emphasizes limb corridor using smooth spatial weighting |
| Segmented output | Suppresses irrelevant background while retaining injury-candidate region |
| CLIP/CLIP-LoRA prediction | Produces ranked class probabilities for final interpretation |

### 6.3 Quantitative Benchmark Status

| Metric Family | Status in This Report | Reason |
|---|---|---|
| Accuracy / Precision / Recall / F1 | Not claimed as final benchmark | No formal labeled evaluation protocol used |
| ROC-AUC / Confusion Matrix | Not claimed as final benchmark | No controlled test benchmark in this run |
| Inference output probabilities | Reported | Available directly from model inference JSON |

Note:
- If required by your course template, you can still include the existing metric assets as demonstration-only outputs, clearly marked non-benchmark.

## 7. Graphs and Tables (Compulsory)

### 7.1 Available Graphs from Project Artifacts

#### Class Distribution
![Class Distribution](Results/report_assets_positive/class_distribution.png)

#### Metric Summary Bar Plot
![Metric Summary](Results/report_assets_positive/metrics_bar.png)

#### ROC One-vs-Rest
![ROC OvR](Results/report_assets_positive/roc_ovr.png)

#### Confusion Matrix Heatmap
![Confusion Matrix](Results/report_assets_positive/confusion_matrix.png)

### 7.2 Prediction Samples / Panels

#### Group Panel
![Sample Group Panel](Results/report_assets_positive/sample_group_panel.png)

#### Additional Prediction Plot
![Sample Predictions](Results/report_assets_positive/sample_predictions.png)

### 7.3 Additional Technical Table: Module-wise Inputs/Outputs

| Module | Input | Processing | Output |
|---|---|---|---|
| Pose Estimation | RGB image | YOLOv8 keypoint inference | 17 body keypoints |
| Limb Selection | Keypoints | Select arm/leg triplet indices | Ordered limb points |
| Sigma Estimation | Adjacent limb points | Distance-based sigma computation | Gaussian spread values |
| Gaussian Splatting | Image + points + sigma | Weighted accumulation + midpoint coverage | Soft attention map |
| Masked Segmentation | Original image + map | Binary mask and bitwise extraction | Region-focused image |
| CLIP/CLIP-LoRA Inference | Segmented image + prompts | Embedding similarity + softmax | Class probabilities |

## 8. Screenshots (Compulsory)
This section lists evidence screenshots/visual outputs available in the workspace.

### 8.1 Output Screenshots Available
- Segmented output image: Results/splatted_output.jpg
- Additional splatted frame: Results/splatted_frame_0028.jpg
- CLIP few-shot visual artifact: CLIP-LoRA/few_shot.png
- Full report plots: Results/report_assets_positive/*.png
- Notebook 03 limb table output: `notebooks/03_yolo_gaussian_lora_clip.ipynb` with columns `Limb`, `Predicted Class`, `Binary Label`, `Injury Score`, `No-Injury Prob`

### 8.2 Recommended Program Proof Captures
Capture and include screenshots for final submission from:
- Pose + keypoint code: pose_gaussian_only.py
- End-to-end script: run_pose_then_clip.py
- CLIP-LoRA inference code: CLIP-LoRA/infer_single_wound.py
- Notebook 03 table output: notebooks/03_yolo_gaussian_lora_clip.ipynb
- Metrics JSON output: Results/report_assets_positive/metrics_summary.json
- Terminal run logs for prediction/evaluation commands

## 9. Student Contribution
Use the following structure and replace with actual names/roll numbers.

| Student Name | Roll No. | Contribution |
|---|---|---|
| Student 1 | XXXX | Pose keypoint extraction, limb selection logic |
| Student 2 | XXXX | Gaussian splatting and segmentation mask generation |
| Student 3 | XXXX | CLIP-LoRA integration, prompt-based inference |
| Student 4 | XXXX | Evaluation plots, report preparation, documentation |

## 10. Discussion
Strengths:
- Interpretable region focus via keypoints and Gaussian weighting.
- Works with pretrained classification models without custom retraining.
- Lightweight adaptation path available using pretrained LoRA.

Limitations:
- No formal benchmark training/evaluation protocol in the current run.
- Performance can vary with pose failure cases, occlusions, and low-resolution imagery.
- Limb-specific heuristic may miss injury in non-selected body parts.

Future work:
- Multi-limb automatic selection based on pose confidence.
- Temporal consistency for video streams.
- Real segmentation supervision and pixel-level IoU evaluation.
- Curated labeled benchmark creation and stratified cross-validation.

## 11. References (Compulsory)
Related works were searched through Google Scholar queries and cross-checked with official paper/project pages.

| Title | Author(s) | Year | Link |
|---|---|---:|---|
| OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields | Zhe Cao, Tomas Simon, Shih-En Wei, Yaser Sheikh | 2019 | https://scholar.google.com/scholar?q=OpenPose+Realtime+Multi-Person+2D+Pose+Estimation+using+Part+Affinity+Fields |
| Simple Baselines for Human Pose Estimation and Tracking | Bin Xiao, Haiping Wu, Yichen Wei | 2018 | https://scholar.google.com/scholar?q=Simple+Baselines+for+Human+Pose+Estimation+and+Tracking |
| Deep High-Resolution Representation Learning for Human Pose Estimation (HRNet) | Ke Sun, Bin Xiao, Dong Liu, Jingdong Wang | 2019 | https://scholar.google.com/scholar?q=Deep+High-Resolution+Representation+Learning+for+Human+Pose+Estimation |
| 3D Gaussian Splatting for Real-Time Radiance Field Rendering | Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, George Drettakis | 2023 | https://scholar.google.com/scholar?q=3D+Gaussian+Splatting+for+Real-Time+Radiance+Field+Rendering |
| Learning Transferable Visual Models From Natural Language Supervision (CLIP) | Alec Radford et al. | 2021 | https://scholar.google.com/scholar?q=Learning+Transferable+Visual+Models+From+Natural+Language+Supervision |
| LoRA: Low-Rank Adaptation of Large Language Models | Edward J. Hu et al. | 2021 | https://scholar.google.com/scholar?q=LoRA+Low-Rank+Adaptation+of+Large+Language+Models |
| CoOp: Learning to Prompt for Vision-Language Models | Kaiyang Zhou et al. | 2022 | https://scholar.google.com/scholar?q=Learning+to+Prompt+for+Vision-Language+Models |
| Tip-Adapter: Training-free CLIP-Adapter for Better Vision-Language Modeling | Renrui Zhang et al. | 2022 | https://scholar.google.com/scholar?q=Tip-Adapter+Training-free+CLIP-Adapter+for+Better+Vision-Language+Modeling |

## 12. Conclusion
The implemented system demonstrates a practical injury-analysis workflow where keypoint geometry guides Gaussian segmentation and pretrained CLIP/CLIP-LoRA performs semantic severity prediction. The report is intentionally presented as an inference-focused implementation without custom classifier training. With a curated labeled benchmark and controlled evaluation protocol, the same architecture can be extended to publishable quantitative validation for field robotics and emergency response.
