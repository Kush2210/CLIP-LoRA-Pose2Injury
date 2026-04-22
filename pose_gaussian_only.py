import argparse
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def limb_selection(limb_name: str):
    mapping = {
        "right_arm": [6, 8, 10],
        "left_arm": [5, 7, 9],
        "right_leg": [12, 14, 16],
        "left_leg": [11, 13, 15],
    }
    return mapping.get(limb_name)


def calculate_sigma(p1, p2, k=0.15):
    p1_tensor = torch.tensor(p1, dtype=torch.float32)
    p2_tensor = torch.tensor(p2, dtype=torch.float32)
    distance = torch.sqrt((p2_tensor[0] - p1_tensor[0]) ** 2 + (p2_tensor[1] - p1_tensor[1]) ** 2)
    sigma = max(float(k * distance), 1.0)
    return sigma


def gaussian_grid(radius, sigma):
    x = np.arange(-radius, radius + 1)
    y = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(x, y)
    g = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    g_sum = g.sum()
    return g / g_sum if g_sum > 0 else g


def apply_gaussian_splatting_image(image, points, sigmas):
    height, width = image.shape[:2]
    splatted = np.zeros_like(image, dtype=np.float32)

    # Add original points and midpoints for smoother limb coverage.
    all_points = list(points)
    for i in range(len(points) - 1):
        mid = (
            int((points[i][0] + points[i + 1][0]) / 2),
            int((points[i][1] + points[i + 1][1]) / 2),
        )
        all_points.append(mid)

    for sigma in sigmas:
        radius = max(int(15 * sigma), 2)
        weights = gaussian_grid(radius, sigma)

        for px, py in all_points:
            px, py = int(px), int(py)

            x_min = max(0, px - radius)
            x_max = min(width, px + radius + 1)
            y_min = max(0, py - radius)
            y_max = min(height, py + radius + 1)

            if x_min >= x_max or y_min >= y_max:
                continue

            gw_x_min = radius - (px - x_min)
            gw_x_max = radius + (x_max - px)
            gw_y_min = radius - (py - y_min)
            gw_y_max = radius + (y_max - py)

            roi_img = image[y_min:y_max, x_min:x_max].astype(np.float32)
            roi_w = weights[gw_y_min:gw_y_max, gw_x_min:gw_x_max][..., None].astype(np.float32)
            splatted[y_min:y_max, x_min:x_max] += roi_img * roi_w

    max_val = splatted.max()
    if max_val > 0:
        splatted = (splatted / max_val) * 255.0
    return splatted.astype(np.uint8)


def get_limb_points(image_path, model, limb_indices):
    results = model.predict(image_path, conf=0.5, verbose=False)
    if not results:
        return None

    r = results[0]
    if r.keypoints is None or r.keypoints.xy is None or r.keypoints.xy.numel() == 0:
        return None

    # Use the first detected person.
    person_kpts = r.keypoints.xy[0].cpu().numpy()
    points = []
    for idx in limb_indices:
        x, y = person_kpts[idx]
        points.append((int(x), int(y)))
    return points


def run_pose_and_splat(input_image, output_image, model_name="yolov8n-pose.pt", limb="right_leg"):
    image = cv2.imread(input_image)
    if image is None:
        raise ValueError(f"Could not read image: {input_image}")

    limb_indices = limb_selection(limb)
    if limb_indices is None:
        raise ValueError("Invalid limb. Use one of: right_arm, left_arm, right_leg, left_leg")

    model = YOLO(model_name)
    points = get_limb_points(input_image, model, limb_indices)
    if not points:
        raise ValueError("No person keypoints detected in the image.")

    sigmas = [calculate_sigma(points[i], points[i + 1]) for i in range(len(points) - 1)]
    splatted = apply_gaussian_splatting_image(image, points, sigmas)

    # Keep only splatted region from original image.
    mask = cv2.inRange(splatted, np.array([1, 1, 1]), np.array([255, 255, 255]))
    result = cv2.bitwise_and(image, image, mask=mask)

    output_dir = os.path.dirname(output_image)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    ok = cv2.imwrite(output_image, result)
    if not ok:
        raise RuntimeError(
            f"Failed to save output image: {output_image}. "
            "Check that the extension is valid (e.g., .jpg, .png)."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pose estimation + Gaussian splatting on one image")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to output image")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="YOLO pose model (default: yolov8n-pose.pt)")
    parser.add_argument(
        "--limb",
        default="right_leg",
        choices=["right_arm", "left_arm", "right_leg", "left_leg"],
        help="Limb to process",
    )
    args = parser.parse_args()

    run_pose_and_splat(args.input, args.output, args.model, args.limb)
    print(f"Saved: {args.output}")
