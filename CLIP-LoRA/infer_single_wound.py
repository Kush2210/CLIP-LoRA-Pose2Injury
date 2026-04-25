import argparse
import json
from types import SimpleNamespace
from pathlib import Path

import torch
from PIL import Image

import clip
from datasets.tromnet import imagenet_classes, imagenet_templates
from loralib.utils import apply_lora, load_lora

CLIP_DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "ViT-B-16.pt"


def resolve_clip_backbone(backbone: str) -> str:
    if backbone == "ViT-B/16" and CLIP_DEFAULT_WEIGHTS.exists():
        return str(CLIP_DEFAULT_WEIGHTS)
    return backbone


def build_text_features(clip_model, device, class_names, prompt_template):
    prompts = [prompt_template.format(classname.replace("_", " ")) for classname in class_names]
    tokens = clip.tokenize(prompts).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features


def predict_single_image(image_path, clip_model, preprocess, text_features, device):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ text_features.t()
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    return probs


def main():
    parser = argparse.ArgumentParser(description="Single-image wound estimation with CLIP + LoRA")
    parser.add_argument("--image", required=True, help="Path to one input image")
    parser.add_argument("--backbone", default="ViT-B/16", type=str)
    parser.add_argument(
        "--class_names",
        nargs="+",
        default=None,
        help="Space-separated class names (default: injury no_injury)",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=None,
        help="Prompt template for classes, must contain {} (default: dataset template)",
    )

    # LoRA loading params (must match training config)
    parser.add_argument("--use_lora", action="store_true", help="Load LoRA weights before inference")
    parser.add_argument("--save_path", type=str, default=None, help="Base folder containing LoRA checkpoints, or a direct .pt checkpoint path")
    parser.add_argument("--filename", type=str, default="lora_weights")
    parser.add_argument("--dataset", type=str, default="tromnet")
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--position", type=str, default="all", choices=["bottom", "mid", "up", "half-up", "half-bottom", "all", "top3"])
    parser.add_argument("--encoder", type=str, default="both", choices=["text", "vision", "both"])
    parser.add_argument("--params", metavar="N", type=str, nargs="+", default=["q", "k", "v"])
    parser.add_argument("--r", type=int, default=2)
    parser.add_argument("--alpha", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.25)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load(resolve_clip_backbone(args.backbone), device=device)
    clip_model.eval()

    class_names = args.class_names if args.class_names else imagenet_classes
    prompt_template = args.prompt_template if args.prompt_template else imagenet_templates[0]
    if "{}" not in prompt_template:
        raise ValueError("--prompt_template must contain {} placeholder")

    if args.use_lora:
        if not args.save_path:
            raise ValueError("--save_path is required when --use_lora is enabled")

        lora_args = SimpleNamespace(
            backbone=args.backbone,
            dataset=args.dataset,
            shots=args.shots,
            seed=args.seed,
            position=args.position,
            encoder=args.encoder,
            params=args.params,
            r=args.r,
            alpha=args.alpha,
            dropout_rate=args.dropout_rate,
            save_path=args.save_path,
            filename=args.filename,
        )
        lora_layers = apply_lora(lora_args, clip_model)
        load_lora(lora_args, lora_layers)
    else:
        print("Warning: running without LoRA weights (--use_lora not set). Predictions may be low quality.")

    text_features = build_text_features(clip_model, device, class_names, prompt_template)
    probs = predict_single_image(args.image, clip_model, preprocess, text_features, device)

    top_idx = int(torch.argmax(probs).item())
    topk = min(3, len(class_names))
    top_vals, top_indices = torch.topk(probs, k=topk)
    result = {
        "image": args.image,
        "predicted_class": class_names[top_idx],
        "probabilities": {cls_name: float(probs[i].item()) for i, cls_name in enumerate(class_names)},
        "top_predictions": [
            {"class": class_names[int(i.item())], "prob": float(v.item())}
            for v, i in zip(top_vals, top_indices)
        ],
        "prompt_template": prompt_template,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
