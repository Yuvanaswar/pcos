"""
Script to construct and execute notebooks/03_COMMON_PREPROCESSING.ipynb
Fulfills all requirements of Part C - Common Preprocessing.
"""

import os
import sys
from pathlib import Path
from src.nb_utils import NotebookBuilder

def build_03_notebook():
    nb = NotebookBuilder(title="03_COMMON_PREPROCESSING: Standardized Letterbox Pipeline")

    # Cell 1: Markdown
    nb.add_markdown(r"""# 03. COMMON PREPROCESSING: Standardized Letterbox Pipeline
## Clinical Ultrasound Preprocessing, Caliper Inpainting, and Augmentation Protocols

### Critical Clinical Constraints:
1. **Aspect-Ratio Preservation:** Never stretch ultrasound images. Follicle circularity and morphology are essential for distinguishing small polycystic antral follicles ($\le 9$ mm) from maturing dominant follicles ($\ge 10$ mm).
2. **Deterministic Validation/Test:** Zero random augmentation on validation or locked-test partitions.
3. **Conservative Augmentation:** Training augmentation is restricted to:
   - Random horizontal flip ($p=0.5$)
   - Mild probe rotation ($\pm 7^\circ$)
   - Mild acoustic gain variation ($\pm 10\%$)
4. **Electronic Caliper Removal:** OpenCV Telea fast-marching inpainting suppresses bright measurement markers to eliminate model reliance on artificial measurement cues.
""")

    # Cell 2: Code - Pipeline Definition & Verification
    nb.add_code(r"""# Cell 1: Pipeline Imports and Verification
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from PIL import Image

from src.preprocessing import (
    letterbox_image,
    inpaint_calipers,
    UltrasoundPreprocessingPipeline,
    IMAGENET_MEAN,
    IMAGENET_STD
)
from src.dataset import OvarianUltrasoundDataset, get_dataloaders

manifest_path = "manifest.csv" if os.path.exists("manifest.csv") else "results/manifest.csv"
df = pd.read_csv(manifest_path)
print(f"[Manifest Loaded] Total entries: {len(df)}")
""")

    # Cell 3: Code - Step-by-Step Visualization of Preprocessing Pipeline
    nb.add_code(r"""# Cell 2: Step-by-step Transformation Pipeline Visualization
sample_row = df[df["class_name"] == "Dominant Follicle"].iloc[0]
sample_path = sample_row["image_path"]

# 1. Original
orig_bgr = cv2.imread(sample_path)
orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
orig_h, orig_w = orig_rgb.shape[:2]

# 2. Caliper Inpainting
inpainted_rgb = inpaint_calipers(orig_rgb)

# 3. Aspect-preserving resize
target_w, target_h = 224, 224
scale = min(target_w / orig_w, target_h / orig_h)
new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
resized_rgb = cv2.resize(inpainted_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

# 4. Letterboxed
letterboxed_rgb, _, _ = letterbox_image(inpainted_rgb, target_size=(224, 224))

# 5. Final Model Input (Normalized Tensor denormalized for visualization)
transform_pipeline = UltrasoundPreprocessingPipeline(target_size=(224, 224), is_train=False)
tensor_out = transform_pipeline(sample_path)  # (3, 224, 224)

# Denormalize for display
denorm = tensor_out.clone().permute(1, 2, 0).numpy()
denorm = (denorm * np.array(IMAGENET_STD)) + np.array(IMAGENET_MEAN)
denorm = np.clip(denorm, 0.0, 1.0)

fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

axes[0].imshow(orig_rgb)
axes[0].set_title(f"1. Original Ultrasound\n({orig_w}x{orig_h})", fontsize=10, fontweight="bold")
axes[0].axis("off")

axes[1].imshow(inpainted_rgb)
axes[1].set_title("2. Caliper Inpainting\n(Telea radius=3)", fontsize=10, fontweight="bold")
axes[1].axis("off")

axes[2].imshow(resized_rgb)
axes[2].set_title(f"3. Resized (Aspect Preserved)\n({new_w}x{new_h})", fontsize=10, fontweight="bold")
axes[2].axis("off")

axes[3].imshow(letterboxed_rgb)
axes[3].set_title("4. Letterbox Padded\n(224x224)", fontsize=10, fontweight="bold")
axes[3].axis("off")

axes[4].imshow(denorm)
axes[4].set_title("5. Final Model Input\n(Normalized Tensor)", fontsize=10, fontweight="bold")
axes[4].axis("off")

plt.tight_layout()
os.makedirs("results/figures", exist_ok=True)
fig_save_path = "results/figures/preprocessing_pipeline.png"
plt.savefig(fig_save_path, dpi=200)
plt.close()
print(f"✓ Saved preprocessing pipeline visualization to: {fig_save_path}")
""")

    # Cell 4: Code - Comparison: Aspect Preserving vs Naive Stretch
    nb.add_code(r"""# Cell 3: Demonstration: Aspect-Ratio Preserving vs Naive Stretching
naive_stretched = cv2.resize(orig_rgb, (224, 224))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(letterboxed_rgb)
axes[0].set_title("Aspect-Ratio Preserved (Letterbox 224x224)\n✓ Anatomical Follicle Geometry Maintained", fontsize=10, color="green", fontweight="bold")
axes[0].axis("off")

axes[1].imshow(naive_stretched)
axes[1].set_title("Naive Stretched (224x224)\n✗ Geometric Distortion & Artificial Eccentricity", fontsize=10, color="red", fontweight="bold")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("results/figures/aspect_ratio_comparison.png", dpi=200)
plt.close()
print("✓ Saved aspect ratio distortion comparison to: results/figures/aspect_ratio_comparison.png")
""")

    # Cell 5: Code - Training Augmentation Demonstration
    nb.add_code(r"""# Cell 4: Training Augmentation Demonstration (Conservative & Clinical)
train_transform = UltrasoundPreprocessingPipeline(target_size=(224, 224), is_train=True)

fig, axes = plt.subplots(1, 6, figsize=(18, 3.5))

for i in range(6):
    aug_tensor = train_transform(sample_path)
    aug_disp = aug_tensor.permute(1, 2, 0).numpy()
    aug_disp = (aug_disp * np.array(IMAGENET_STD)) + np.array(IMAGENET_MEAN)
    aug_disp = np.clip(aug_disp, 0.0, 1.0)
    
    axes[i].imshow(aug_disp)
    axes[i].set_title(f"Train Aug #{i+1}", fontsize=10)
    axes[i].axis("off")

plt.suptitle("Conservative Training Augmentations (Flip p=0.5, Rot ±7°, Gain ±10%)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("results/figures/augmentation_samples.png", dpi=200)
plt.close()
print("✓ Saved training augmentation examples to: results/figures/augmentation_samples.png")
""")

    # Cell 6: Code - DataLoader Verification & Batch Shapes
    nb.add_code(r"""# Cell 5: DataLoader Batch Verification & Class Balancing Check
train_loader, val_loader, test_loader, class_weights = get_dataloaders(
    manifest_path=manifest_path,
    batch_size=16,
    num_workers=0,
    use_sampler=True
)

print("DataLoader Summary:")
print(f"  Train Batches:      {len(train_loader)} (Batch size = 16, Total = {len(train_loader.dataset)})")
print(f"  Validation Batches: {len(val_loader)} (Batch size = 16, Total = {len(val_loader.dataset)})")
print(f"  Test Batches:       {len(test_loader)} (Batch size = 16, Total = {len(test_loader.dataset)})")
print(f"  Class Weights:      {class_weights.tolist()}")

batch_x, batch_y, batch_paths = next(iter(train_loader))
print(f"\nBatch Tensor Verification:")
print(f"  Input Tensor Shape: {batch_x.shape} (Expected: [16, 3, 224, 224])")
print(f"  Labels Tensor:      {batch_y.tolist()}")
print(f"  Tensor Dtype:       {batch_x.dtype}")
print(f"  Value Range:        Min = {batch_x.min():.3f}, Max = {batch_x.max():.3f}")

# Verification assertions
assert batch_x.shape == (16, 3, 224, 224), f"Unexpected shape {batch_x.shape}"
assert len(batch_y) == 16, f"Unexpected label batch length {len(batch_y)}"
print("\n✓ All preprocessing assertions PASSED. Pipeline ready for multi-model benchmark training!")
""")

    out_path = "notebooks/03_COMMON_PREPROCESSING.ipynb"
    nb.execute_and_save(out_path, working_dir=".")
    print(f"Successfully generated and executed {out_path}!")

if __name__ == "__main__":
    build_03_notebook()
