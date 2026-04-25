import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


POSITIVE_LABEL = "injury"
NEGATIVE_LABEL = "no_injury"


def load_rows(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")

    return rows


def pick_label(row, *keys):
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value.strip()
    raise KeyError(f"None of the label keys were present: {keys}")


def binary_metrics(y_true, y_pred, positive_label=POSITIVE_LABEL):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("No labels provided")

    tp = fp = tn = fn = 0
    for truth, pred in zip(y_true, y_pred):
        truth_pos = truth == positive_label
        pred_pos = pred == positive_label
        if truth_pos and pred_pos:
            tp += 1
        elif not truth_pos and pred_pos:
            fp += 1
        elif not truth_pos and not pred_pos:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "support": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def confusion_matrix_counts(y_true, y_pred, positive_label=POSITIVE_LABEL):
    tp = fp = tn = fn = 0
    for truth, pred in zip(y_true, y_pred):
        truth_pos = truth == positive_label
        pred_pos = pred == positive_label
        if truth_pos and pred_pos:
            tp += 1
        elif not truth_pos and pred_pos:
            fp += 1
        elif not truth_pos and not pred_pos:
            tn += 1
        else:
            fn += 1
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def probability_metrics(rows):
    if not rows:
        raise ValueError("No rows provided")

    total = len(rows)
    true_class_prob_sum = 0.0
    prob_error_sum = 0.0
    brier_sum = 0.0
    margin_sum = 0.0

    for row in rows:
        truth = pick_label(row, "ground_truth", "gt_binary", "gt_label")
        injury_prob = float(row["injury_prob"])
        no_injury_prob = float(row["no_injury_prob"])
        if truth == POSITIVE_LABEL:
            true_class_prob = injury_prob
            prob_error = 1.0 - injury_prob
            brier = (1.0 - injury_prob) ** 2 + (0.0 - no_injury_prob) ** 2
        else:
            true_class_prob = no_injury_prob
            prob_error = 1.0 - no_injury_prob
            brier = (0.0 - injury_prob) ** 2 + (1.0 - no_injury_prob) ** 2

        true_class_prob_sum += true_class_prob
        prob_error_sum += prob_error
        brier_sum += brier
        margin_sum += abs(injury_prob - no_injury_prob)

    return {
        "mean_true_class_prob": true_class_prob_sum / total,
        "mean_probability_error": prob_error_sum / total,
        "mean_margin": margin_sum / total,
        "mean_brier": brier_sum / total,
    }


def print_metrics(title, metrics):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 score : {metrics['f1']:.4f}")
    print(f"TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']} Support={metrics['support']}")
    if "mean_true_class_prob" in metrics:
        print(f"Mean true-class prob   : {metrics['mean_true_class_prob']:.4f}")
        print(f"Mean probability error : {metrics['mean_probability_error']:.4f}")
        print(f"Mean abs margin        : {metrics['mean_margin']:.4f}")
        print(f"Mean Brier score       : {metrics['mean_brier']:.4f}")


def save_confusion_matrix(confusion, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "confusion_matrix.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", "pred_no_injury", "pred_injury"])
        writer.writerow(["actual_no_injury", int(confusion[0, 0]), int(confusion[0, 1])])
        writer.writerow(["actual_injury", int(confusion[1, 0]), int(confusion[1, 1])])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks([0, 1], labels=["no_injury", "injury"])
    ax.set_yticks([0, 1], labels=["no_injury", "injury"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(j, i, str(int(confusion[i, j])), ha="center", va="center", color="black", fontsize=12)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    png_path = output_dir / "confusion_matrix.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Confusion matrix CSV written to {csv_path}")
    print(f"Confusion matrix PNG written to {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze batch report results.")
    parser.add_argument(
        "--comparison_csv",
        default=str(Path("Results") / "batch_reports" / "ground_truth_vs_predicted.csv"),
        help="Per-limb comparison CSV written by batch_testing_report.py",
    )
    args = parser.parse_args()

    comparison_rows = load_rows(Path(args.comparison_csv))
    y_true = [pick_label(row, "ground_truth", "gt_binary", "gt_label") for row in comparison_rows]
    y_pred = []
    for row in comparison_rows:
        injury_prob = float(row["injury_prob"])
        no_injury_prob = float(row["no_injury_prob"])
        y_pred.append(POSITIVE_LABEL if injury_prob >= no_injury_prob else NEGATIVE_LABEL)

    limb_metrics = binary_metrics(y_true, y_pred)
    limb_metrics.update(probability_metrics(comparison_rows))
    print_metrics("Limb-level metrics", limb_metrics)

    confusion = confusion_matrix_counts(y_true, y_pred)
    print("\nConfusion matrix")
    print(confusion)
    save_confusion_matrix(confusion, Path("Results") / "batch_reports")


if __name__ == "__main__":
    main()
