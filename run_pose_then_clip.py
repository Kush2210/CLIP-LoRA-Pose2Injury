import argparse
import os
import subprocess
import sys
from typing import Optional

from pose_gaussian_only import run_pose_and_splat


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CLIP_LORA_DIR = os.path.join(PROJECT_ROOT, "CLIP-LoRA")
INFER_SCRIPT = os.path.join(CLIP_LORA_DIR, "infer_single_wound.py")
MAIN_SCRIPT = os.path.join(CLIP_LORA_DIR, "main.py")


def run_command(command):
    process = subprocess.run(command, cwd=PROJECT_ROOT)
    if process.returncode != 0:
        raise SystemExit(process.returncode)


def resolve_python(python_executable: Optional[str]):
    if python_executable:
        return python_executable
    return sys.executable


def main():
    parser = argparse.ArgumentParser(description="Run pose Gaussian splatting first, then CLIP-LoRA")
    parser.add_argument("--input", required=True, help="Path to the original input image")
    parser.add_argument("--splatted_output", default="Results/splatted_output.jpg", help="Path to save the splatted image")
    parser.add_argument("--pose_model", default="yolov8n-pose.pt", help="YOLO pose model path/name")
    parser.add_argument("--limb", default="right_leg", choices=["right_arm", "left_arm", "right_leg", "left_leg"])

    parser.add_argument("--mode", default="predict", choices=["predict", "evaluate"], help="predict = single image, evaluate = CLIP-LoRA main.py")

    parser.add_argument("--clip_backbone", default="ViT-B/16")
    parser.add_argument(
        "--class_names",
        metavar="CLS",
        type=str,
        nargs="+",
        default=None,
        help="Space-separated class names for CLIP inference",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=None,
        help="Prompt template for classes (must include {})",
    )
    parser.add_argument("--use_lora", action="store_true", help="Load LoRA weights before CLIP inference")
    parser.add_argument("--lora_save_path", default=None, help="Base folder containing LoRA checkpoints")
    parser.add_argument("--filename", default="lora_weights")
    parser.add_argument("--dataset", default="tromnet")
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--position", default="all", choices=["bottom", "mid", "up", "half-up", "half-bottom", "all", "top3"])
    parser.add_argument("--encoder", default="both", choices=["text", "vision", "both"])
    parser.add_argument("--params", metavar="N", type=str, nargs="+", default=["q", "k", "v"])
    parser.add_argument("--r", type=int, default=2)
    parser.add_argument("--alpha", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.25)

    parser.add_argument("--evaluate_root_path", default=None, help="Dataset root for CLIP-LoRA main.py evaluation mode")
    parser.add_argument("--evaluate_save_path", default=None, help="LoRA checkpoint base path for evaluation mode")
    parser.add_argument(
        "--python_executable",
        default=None,
        help="Python executable used to launch CLIP-LoRA scripts (default: current interpreter)",
    )

    args = parser.parse_args()
    python_exec = resolve_python(args.python_executable)

    # Step 1: pose estimation + gaussian splatting
    run_pose_and_splat(args.input, args.splatted_output, args.pose_model, args.limb)
    print(f"Saved splatted image: {args.splatted_output}")

    # Step 2: CLIP-LoRA
    if args.mode == "predict":
        command = [
            python_exec,
            INFER_SCRIPT,
            "--image",
            args.splatted_output,
            "--backbone",
            args.clip_backbone,
        ]
        if args.class_names:
            command.extend(["--class_names", *args.class_names])
        if args.prompt_template:
            command.extend(["--prompt_template", args.prompt_template])
        if args.use_lora:
            if not args.lora_save_path:
                raise ValueError("--lora_save_path is required when --use_lora is enabled")
            command.extend([
                "--use_lora",
                "--save_path",
                args.lora_save_path,
                "--filename",
                args.filename,
                "--dataset",
                args.dataset,
                "--shots",
                str(args.shots),
                "--seed",
                str(args.seed),
                "--position",
                args.position,
                "--encoder",
                args.encoder,
                "--params",
                *args.params,
                "--r",
                str(args.r),
                "--alpha",
                str(args.alpha),
                "--dropout_rate",
                str(args.dropout_rate),
            ])

        print("Running CLIP-LoRA single-image inference...")
        run_command(command)
        return

    # Dataset evaluation mode for CLIP-LoRA main.py
    if not args.evaluate_root_path:
        raise ValueError("--evaluate_root_path is required when --mode evaluate")
    if not args.evaluate_save_path:
        raise ValueError("--evaluate_save_path is required when --mode evaluate")

    command = [
        python_exec,
        MAIN_SCRIPT,
        "--root_path",
        args.evaluate_root_path,
        "--backbone",
        args.clip_backbone,
        "--save_path",
        args.evaluate_save_path,
        "--filename",
        args.filename,
        "--dataset",
        args.dataset,
        "--shots",
        str(args.shots),
        "--seed",
        str(args.seed),
        "--position",
        args.position,
        "--encoder",
        args.encoder,
        "--params",
        *args.params,
        "--r",
        str(args.r),
        "--alpha",
        str(args.alpha),
        "--dropout_rate",
        str(args.dropout_rate),
    ]

    if args.use_lora:
        command.append("--eval_only")

    print("Running CLIP-LoRA evaluation...")
    run_command(command)


if __name__ == "__main__":
    main()
