import os
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry


def keep_largest_component(binary_mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return binary_mask
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_idx).astype(np.uint8)


def clean_mask(binary_mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    return keep_largest_component(closed)


base_dir = Path(__file__).resolve().parent
image_path = base_dir / "sofa.jpg"
checkpoint_path = base_dir / "checkpoints" / "sam_vit_b_01ec64.pth"
mask_path = base_dir / "mask.png"
overlay_path = base_dir / "segmentation_result.png"

image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("Image loaded")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint_path))
sam.to(device=device)
predictor = SamPredictor(sam)
print("SAM model loaded")

predictor.set_image(image_rgb)

h, w = image.shape[:2]
input_box = np.array([int(w * 0.1), int(h * 0.2), int(w * 0.9), int(h * 0.8)])
print("Bounding box:", input_box.tolist())

with torch.no_grad():
    masks, scores, _ = predictor.predict(box=input_box, multimask_output=True)

best_idx = int(np.argmax(scores))
best_score = float(scores[best_idx])
mask = masks[best_idx].astype(np.uint8)
mask = clean_mask(mask)
mask_u8 = (mask * 255).astype(np.uint8)
print(f"Segmentation done. Best score: {best_score:.4f}")

cv2.imwrite(str(mask_path), mask_u8)

overlay = image.copy()
overlay[mask > 0] = (0, 255, 0)
result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
cv2.imwrite(str(overlay_path), result)

print("Saved:")
print(mask_path)
print(overlay_path)