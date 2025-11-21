import os
from pathlib import Path

import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from transformers import Sam3Processor, Sam3Model
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3").to(device).eval()

image_dir = Path("/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/xview/processed/split/test/images")

# Take last 15 images
image_paths = sorted([p for p in image_dir.glob("*.png")])
print("Using images:")
for p in image_paths:
    print("  ", p.name)

# Directory to save labels
print("parent directory:", image_dir.parent)
labels_dir = image_dir.parent / "created_labels_by_sam"
labels_dir.mkdir(parents=True, exist_ok=True)
print("Saving labels to:", labels_dir)

concepts = ["airplane", "storage tank", "ship or boat"]
colors = ["r", "g", "b", "y", "c", "m"]

model.eval()

for img_path in image_paths:
    img = Image.open(img_path).convert("RGB")

    all_boxes = []  # (box_tensor, score, concept, color)

    for ci, concept in enumerate(concepts):
        inputs = processor(images=img, text=concept, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs["original_sizes"].tolist(),
        )[0]

        boxes = results["boxes"]       # [N, 4] xyxy
        scores = results["scores"]     # [N]

        for b, s in zip(boxes, scores):
            all_boxes.append(
                (b.cpu(), float(s), concept, colors[ci % len(colors)])
            )

    # 1) Save labels as .txt: one line per box
    # format: concept score x0 y0 x1 y1  (pixel coordinates)
    label_path = labels_dir / f"{img_path.stem}.txt"
    with open(label_path, "w") as f:
        for box, score, concept, _ in all_boxes:
            x0, y0, x1, y1 = box.tolist()
            f.write(f"{concept} {score:.4f} {x0:.1f} {y0:.1f} {x1:.1f} {y1:.1f}\n")

    # # 2) Plot with all predicted boxes for this image
    # fig, ax = plt.subplots(1, figsize=(8, 8))
    # ax.imshow(img)

    # for box, score, concept, color in all_boxes:
    #     x0, y0, x1, y1 = box.tolist()
    #     w, h = x1 - x0, y1 - y0

    #     rect = patches.Rectangle(
    #         (x0, y0),
    #         w,
    #         h,
    #         linewidth=2,
    #         edgecolor=color,
    #         facecolor="none",
    #     )
    #     ax.add_patch(rect)
    #     ax.text(
    #         x0,
    #         max(y0 - 3, 0),
    #         f"{concept} {score:.2f}",
    #         fontsize=8,
    #         color="white",
    #         bbox=dict(facecolor=color, alpha=0.5, linewidth=0),
    #     )

    # ax.set_title(img_path.name)
    # ax.axis("off")
    # plt.show()