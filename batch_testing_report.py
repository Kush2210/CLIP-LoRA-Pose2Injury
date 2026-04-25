import argparse
import csv
import os
import sys
from pathlib import Path

import certifi
import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from pose_gaussian_only import calculate_sigma, apply_gaussian_splatting_image


PROJECT_ROOT = Path(__file__).resolve().parent
CLIP_LORA_DIR = PROJECT_ROOT / "CLIP-LoRA"
if str(CLIP_LORA_DIR) not in sys.path:
    sys.path.insert(0, str(CLIP_LORA_DIR))

import importlib

clip = importlib.import_module("clip")
loralib_utils = importlib.import_module("loralib.utils")


BODY_PARTS = {
    "right_arm": [6, 8, 10],
    "left_arm": [5, 7, 9],
    "right_leg": [12, 14, 16],
    "left_leg": [11, 13, 15],
}

SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

CLASS_NAMES = ["injury", "no_injury"]
PROMPT_TEMPLATES = [
    "For the highlighted limb {} is present.",
    "The limb condition is {}.",
    "This limb shows {}.",
]
LIMB_LABEL_COLUMNS = ["left_hand", "right_hand", "left_leg", "right_leg"]


def build_clip_model(backbone: str, lora_ckpt: Path, dataset: str = "tromnet", shots: int = 16,
                     seed: int = 1, position: str = "all", encoder: str = "both",
                     params=None, r: int = 2, alpha: int = 1, dropout_rate: float = 0.25):
    if params is None:
        params = ["q", "k", "v"]

    os.environ["SSL_CERT_FILE"] = certifi.where()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        clip_model, preprocess = clip.load(backbone, device=device)
    except Exception as e:
        if "CERTIFICATE_VERIFY_FAILED" in str(e):
            os.environ["PYTHONHTTPSVERIFY"] = "0"
            clip_model, preprocess = clip.load(backbone, device=device)
        else:
            raise

    clip_model.eval()

    if not lora_ckpt.exists():
        raise FileNotFoundError(f"LoRA checkpoint missing: {lora_ckpt}")
    from types import SimpleNamespace

    lora_args = SimpleNamespace(
        backbone=backbone,
        dataset=dataset,
        shots=shots,
        seed=seed,
        position=position,
        encoder=encoder,
        params=params,
        r=r,
        alpha=alpha,
        dropout_rate=dropout_rate,
        save_path=str(lora_ckpt),
        filename="lora_weights",
    )
    lora_layers = loralib_utils.apply_lora(lora_args, clip_model)
    loralib_utils.load_lora(lora_args, lora_layers)

    runtime_info = {
        "device": device,
        "torch_version": torch.__version__,
        "has_sdp": hasattr(torch.nn.functional, "scaled_dot_product_attention"),
        "use_lora": True,
    }

    return device, clip_model, preprocess, runtime_info


def build_text_features(clip_model, device: str, class_names, prompt_templates):
    class_embeddings = []
    with torch.no_grad():
        for cls_name in class_names:
            prompts = [template.format(cls_name.replace("_", " ")) for template in prompt_templates]
            tokens = clip.tokenize(prompts).to(device)
            prompt_features = clip_model.encode_text(tokens)
            prompt_features = prompt_features / prompt_features.norm(dim=-1, keepdim=True)
            cls_feature = prompt_features.mean(dim=0)
            cls_feature = cls_feature / cls_feature.norm()
            class_embeddings.append(cls_feature)

    return torch.stack(class_embeddings, dim=0)


def predict_probs(image_path: Path, clip_model, preprocess, text_features, device: str):
    img = Image.open(image_path).convert("RGB")
    t = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(t)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ text_features.t()
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    return probs.detach().cpu().numpy()


def summarize_probs(probs_lora: np.ndarray):
    pred_idx = int(np.argmax(probs_lora))
    pred_class = CLASS_NAMES[pred_idx]

    injury_idx = CLASS_NAMES.index("injury")
    no_injury_idx = CLASS_NAMES.index("no_injury")

    injury_score = float(probs_lora[injury_idx])
    no_injury_prob = float(probs_lora[no_injury_idx])
    pred_binary = "injury" if injury_score >= no_injury_prob else "no_injury"

    return {
        "predicted_class": pred_class,
        "pred_binary": pred_binary,
        "injury_score": injury_score,
        "injury_margin": injury_score - no_injury_prob,
        "injury_prob": float(probs_lora[injury_idx]),
        "no_injury_prob": no_injury_prob,
    }


def load_ground_truth(ground_truth_csv: Path):
    if not ground_truth_csv.exists():
        raise FileNotFoundError(f"Ground truth CSV missing: {ground_truth_csv}")

    import csv as _csv

    gt_map = {}
    with ground_truth_csv.open("r", newline="") as f:
        reader = _csv.DictReader(f)
        expected = {"image", *LIMB_LABEL_COLUMNS}
        missing = expected - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                f"Ground truth CSV must include columns {sorted(expected)}; missing {sorted(missing)}"
            )
        for row in reader:
            image_name = row["image"].strip()
            gt_map[image_name] = {col: row[col].strip() for col in LIMB_LABEL_COLUMNS}
    return gt_map


def normalize_limb_name(part_name: str) -> str:
    return part_name.replace("arm", "hand")


def get_limb_points(person_kpts: np.ndarray, limb_name: str):
    limb_indices = BODY_PARTS[limb_name]
    return [(int(person_kpts[idx][0]), int(person_kpts[idx][1])) for idx in limb_indices]


def overlay_keypoints(image_rgb: np.ndarray, person_kpts: np.ndarray):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image_rgb)
    for start_idx, end_idx in SKELETON_PAIRS:
        x1, y1 = person_kpts[start_idx]
        x2, y2 = person_kpts[end_idx]
        ax.plot([x1, x2], [y1, y2], linewidth=2, color="cyan", alpha=0.9)
    for idx, (x, y) in enumerate(person_kpts):
        ax.scatter([x], [y], s=20, color="yellow")
        ax.text(x + 2, y + 2, str(idx), fontsize=7, color="white",
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1))
    ax.axis("off")
    return fig, ax


def make_image_report(
    img_path: Path,
    gt_parts: dict,
    pose_model,
    clip_model_lora,
    preprocess_lora,
    text_features_lora,
    device: str,
    output_dir: Path,
    mask_dir: Path,
):
    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = pose_model.predict(str(img_path), conf=0.4, verbose=False)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    if not results or results[0].keypoints is None or results[0].keypoints.xy is None or results[0].keypoints.xy.numel() == 0:
        for ax in axes[1:]:
            ax.axis("off")
        fig.suptitle(f"{img_path.name} | no person detected", fontsize=15, color="crimson")
        output_path = output_dir / f"{img_path.stem}_report.jpg"
        fig.tight_layout()
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return {
            "image": img_path.name,
            "final_label": "no_person",
            "output_path": str(output_path),
            "parts": [],
        }

    person_kpts = results[0].keypoints.xy[0].cpu().numpy()

    overlay_fig, _ = overlay_keypoints(image_rgb, person_kpts)
    overlay_canvas = None
    overlay_fig.canvas.draw()
    overlay_canvas = np.frombuffer(overlay_fig.canvas.tostring_rgb(), dtype=np.uint8)
    overlay_canvas = overlay_canvas.reshape(overlay_fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(overlay_fig)

    axes[1].imshow(overlay_canvas)
    axes[1].set_title("Keypoints + skeleton", fontsize=11)
    axes[1].axis("off")

    limb_results = []
    part_titles = ["right_arm", "left_arm", "right_leg", "left_leg"]
    for ax, part_name in zip(axes[2:], part_titles):
        points = get_limb_points(person_kpts, part_name)
        sigmas = [calculate_sigma(points[i], points[i + 1]) for i in range(len(points) - 1)]
        splatted = apply_gaussian_splatting_image(image_bgr, points, sigmas)
        mask = cv2.inRange(splatted, np.array([1, 1, 1]), np.array([255, 255, 255]))
        masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)

        mask_path = mask_dir / f"{img_path.stem}_{part_name}_gaussian_mask.png"
        cv2.imwrite(str(mask_path), masked)

        probs_lora = predict_probs(mask_path, clip_model_lora, preprocess_lora, text_features_lora, device)
        pred = summarize_probs(probs_lora)
        gt_part_name = normalize_limb_name(part_name)
        gt_label = gt_parts.get(gt_part_name, "no_injury")
        limb_results.append({"part": part_name, "ground_truth": gt_label, **pred, "mask_path": str(mask_path)})

        ax.imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
        gt_binary = "injury" if gt_label == "injury" else "no_injury"
        color = "crimson" if pred["pred_binary"] == "injury" else "green"
        ax.set_title(
            f"{part_name}\nGT: {gt_binary} | Pred: {pred['pred_binary']}\n"
            f"injury_score={pred['injury_score']:.4f}, no_injury_prob={pred['no_injury_prob']:.4f}",
            fontsize=10,
            color=color,
        )
        ax.axis("off")

    final_label = "injury" if any(item["pred_binary"] == "injury" for item in limb_results) else "no_injury"
    final_gt_label = "injury" if any(value == "injury" for value in gt_parts.values()) else "no_injury"
    max_injury_score = max((item["injury_score"] for item in limb_results), default=0.0)
    fig.suptitle(
        f"{img_path.name} | GT: {final_gt_label} | Pred: {final_label} | max injury score: {max_injury_score:.4f}",
        fontsize=16,
        color="crimson" if final_label == "injury" else "green",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = output_dir / f"{img_path.stem}_report.jpg"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "image": img_path.name,
        "final_label": final_label,
        "output_path": str(output_path),
        "parts": limb_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch report over testing images with pose + LoRA CLIP.")
    parser.add_argument("--input_dir", default=str(PROJECT_ROOT / "DATASET" / "images"))
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "Results" / "batch_reports"))
    parser.add_argument("--mask_dir", default=str(PROJECT_ROOT / "Results" / "batch_reports" / "part_masks"))
    parser.add_argument("--ground_truth_csv", default=str(PROJECT_ROOT / "DATASET" / "ground_truth.csv"))
    parser.add_argument("--pose_model", default="yolov8n-pose.pt")
    parser.add_argument("--backbone", default="ViT-B/16")
    parser.add_argument("--lora_ckpt", default=str(CLIP_LORA_DIR / "weights" / "lora_weights_960_2.9566854533582632e-05.pt"))
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    mask_dir = Path(args.mask_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {input_dir}")

    pose_model = YOLO(args.pose_model)
    gt_map = load_ground_truth(Path(args.ground_truth_csv))
    device_lora, clip_model_lora, preprocess_lora, runtime_lora = build_clip_model(
        backbone=args.backbone,
        lora_ckpt=Path(args.lora_ckpt),
    )
    text_features_lora = build_text_features(clip_model_lora, device_lora, CLASS_NAMES, PROMPT_TEMPLATES)

    print(
        "Runtime (clip+lora):",
        f"torch={runtime_lora['torch_version']},",
        f"sdp={runtime_lora['has_sdp']},",
        f"device={runtime_lora['device']}",
    )

    summary_rows = []
    for img_path in image_paths:
        gt_parts = gt_map.get(img_path.name)
        if gt_parts is None:
            raise RuntimeError(f"Missing ground-truth row for {img_path.name}")
        result = make_image_report(
            img_path=img_path,
            gt_parts=gt_parts,
            pose_model=pose_model,
            clip_model_lora=clip_model_lora,
            preprocess_lora=preprocess_lora,
            text_features_lora=text_features_lora,
            device=device_lora,
            output_dir=output_dir,
            mask_dir=mask_dir,
        )
        summary_rows.append(result)
        print(f"Saved {result['output_path']} -> {result['final_label']}")

    gt_rows = []
    for row in summary_rows:
        image_name = row["image"]
        gt_parts = gt_map.get(image_name)

        pred_part_map = {part["part"]: part for part in row["parts"]}
        for pred_part_name, pred_part in pred_part_map.items():
            gt_part_name = normalize_limb_name(pred_part_name)
            if gt_part_name not in gt_parts:
                raise RuntimeError(f"Ground-truth column missing for part {gt_part_name}")
            gt_label = gt_parts[gt_part_name]
            gt_binary = "injury" if gt_label == "injury" else "no_injury"
            gt_rows.append(
                {
                    "image": image_name,
                    "part": gt_part_name,
                    "ground_truth": gt_binary,
                    "predicted_label": pred_part["pred_binary"],
                    "predicted_class": pred_part["predicted_class"],
                    "injury_prob": f"{pred_part['injury_prob']:.8f}",
                    "no_injury_prob": f"{pred_part['no_injury_prob']:.8f}",
                    "injury_score": f"{pred_part['injury_score']:.8f}",
                    "correct": "yes" if gt_binary == pred_part["pred_binary"] else "no",
                    "mask_path": pred_part["mask_path"],
                }
            )

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "final_label", "output_path"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row[k] for k in ["image", "final_label", "output_path"]})

    detailed_csv_path = output_dir / "part_probabilities.csv"
    with detailed_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "part",
                "ground_truth",
                "predicted_label",
                "predicted_class",
                "pred_binary",
                "injury_prob",
                "no_injury_prob",
                "injury_score",
                "correct",
                "mask_path",
            ],
        )
        writer.writeheader()
        for row in gt_rows:
            writer.writerow(row)

    gt_summary_path = output_dir / "ground_truth_vs_predicted.csv"
    with gt_summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "part",
                "ground_truth",
                "predicted_label",
                "predicted_class",
                "injury_prob",
                "no_injury_prob",
                "injury_score",
                "correct",
                "mask_path",
            ],
        )
        writer.writeheader()
        for row in gt_rows:
            writer.writerow(row)

    print(f"Summary written to {csv_path}")
    print(f"Detailed part probabilities written to {detailed_csv_path}")
    print(f"Ground truth comparison written to {gt_summary_path}")


if __name__ == "__main__":
    main()
