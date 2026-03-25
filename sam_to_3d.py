import os
import cv2
import torch
import numpy as np
import open3d as o3d
from transformers import pipeline
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image

# -------------------------
# 1. Papka
# -------------------------

image_folder = "multi_images"

print("✔ Rasm papkasi:", image_folder)

# -------------------------
# 2. Depth model
# -------------------------

print("Depth model yuklanmoqda...")

depth_estimator = pipeline(
    task="depth-estimation",
    model="Intel/dpt-large"
)

print("✔ Depth model tayyor")

# -------------------------
# 3. SAM model
# -------------------------

sam_checkpoint = "checkpoints/sam_vit_b_01ec64.pth"

sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)

sam.to(device="cpu")

predictor = SamPredictor(sam)

print("✔ SAM model yuklandi")

# -------------------------
# 4. Point cloud uchun
# -------------------------

all_points = []
all_colors = []

# -------------------------
# 5. Rasmlarni o‘qish
# -------------------------

for file in os.listdir(image_folder):

    if not file.lower().endswith(".jpg"):
        continue

    image_path = os.path.join(image_folder, file)

    print("\nProcessing:", image_path)

    image = cv2.imread(image_path)

    if image is None:
        print("❌ Rasm topilmadi:", image_path)
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w, _ = image.shape

    print("✔ Rasm yuklandi")

    # -------------------------
    # SAM segmentation
    # -------------------------

    predictor.set_image(image_rgb)

    input_box = np.array([100, 200, w-100, h-100])

    masks, scores, logits = predictor.predict(
        box=input_box,
        multimask_output=False
    )

    mask = masks[0]

    masked_image = image_rgb.copy()
    masked_image[~mask] = 0

    print("✔ Object ajratildi")

    # -------------------------
    # Depth estimation
    # -------------------------

    pil_image = Image.fromarray(masked_image)

    depth_output = depth_estimator(pil_image)

    depth = np.array(depth_output["depth"])

    depth = cv2.resize(depth, (w, h))

    depth = depth / depth.max()

    print("✔ Depth hisoblandi")

    # -------------------------
    # Camera param
    # -------------------------

    fx = fy = 1000
    cx = w / 2
    cy = h / 2

    # -------------------------
    # 3D point cloud
    # -------------------------

    for v in range(h):
        for u in range(w):

            if not mask[v, u]:
                continue

            z = depth[v, u] * 5

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            all_points.append([x, y, z])
            all_colors.append(image_rgb[v, u] / 255.0)

    print("✔ Pointlar qo‘shildi")

# -------------------------
# 6. Point cloud yaratish
# -------------------------

points = np.array(all_points)
colors = np.array(all_colors)

print("\n✔ Umumiy pointlar:", len(points))

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# -------------------------
# 7. 3D model saqlash
# -------------------------

output_file = "multi_view_model.ply"

o3d.io.write_point_cloud(output_file, pcd)

print("✔ 3D model saqlandi:", output_file)