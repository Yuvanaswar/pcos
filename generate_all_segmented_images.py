"""
Generate and export segmented / ROI-extracted images for all 572 ultrasound scans.
Outputs:
  - results/segmented_images/crops/<split>/<class>/<filename> (cropped & letterboxed parenchymal ROI)
  - results/segmented_images/overlays/<split>/<class>/<filename> (original scan + green segmented boundary + bounding box)
  - results/figures/all_segmented_samples.png (multi-class clinical inspection gallery)
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.preprocessing import letterbox_image

def segment_and_crop(img_bgr):
    """
    Performs acoustic parenchymal segmentation:
    - Otsu adaptive thresholding on blurred acoustic intensity
    - Morphological closure to unify ovarian parenchyma
    - Extracts largest parenchymal contour and bounding box with 5% safety margin
    - Returns: (cropped_letterbox_224, overlay_bgr, mask_uint8)
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur to filter high-frequency speckle noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    overlay = img_bgr.copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [c], -1, 255, -1)
        
        # Draw green contour on overlay
        cv2.drawContours(overlay, [c], -1, (0, 255, 0), 2)
        
        # Bounding box with 5% safety margin to protect subcapsular follicles
        x, y, bw, bh = cv2.boundingRect(c)
        pad_x = int(bw * 0.05)
        pad_y = int(bh * 0.05)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w, x + bw + pad_x)
        y1 = min(h, y + bh + pad_y)
        
        # Draw bounding box on overlay in cyan
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 255, 0), 2)
        
        cropped = img_bgr[y0:y1, x0:x1]
        if cropped.size == 0:
            cropped = img_bgr
    else:
        cropped = img_bgr

    letterboxed, _, _ = letterbox_image(cropped, target_size=(224, 224))
    return letterboxed, overlay, mask


def main():
    manifest_path = "manifest.csv" if os.path.exists("manifest.csv") else "results/manifest.csv"
    df = pd.read_csv(manifest_path)
    print(f"Loaded manifest with {len(df)} images.")
    
    base_out = Path("results/segmented_images")
    crops_dir = base_out / "crops"
    overlays_dir = base_out / "overlays"
    
    total = len(df)
    print(f"Segmenting all {total} ultrasound scans...")
    
    for idx, row in df.iterrows():
        p = row["image_path"]
        split = row["split"]
        cname = row["class_name"].replace(" ", "_")
        fname = Path(p).name
        
        img = cv2.imread(p)
        if img is None:
            print(f"Warning: could not read {p}")
            continue
            
        crop, overlay, mask = segment_and_crop(img)
        
        # Output paths
        c_dir = crops_dir / split / cname
        o_dir = overlays_dir / split / cname
        c_dir.mkdir(parents=True, exist_ok=True)
        o_dir.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(c_dir / fname), crop)
        cv2.imwrite(str(o_dir / fname), overlay)
        
        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            print(f"  Processed {idx + 1}/{total} scans ({(idx + 1)/total*100:.1f}%)")

    print("\n[DONE] Successfully exported all segmented crops and overlays!")
    print(f"  Crops folder:    {crops_dir}")
    print(f"  Overlays folder: {overlays_dir}")

    # Generate Representative Multi-Class Comparison Figure
    print("\nGenerating multi-class segmentation gallery figure...")
    classes = ["Normal Ovary", "PCOS", "Dominant Follicle"]
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    
    for row_idx, cls in enumerate(classes):
        sub = df[df["class_name"] == cls]
        sample_p = sub.iloc[0]["image_path"]
        img_bgr = cv2.imread(sample_p)
        crop_bgr, overlay_bgr, mask = segment_and_crop(img_bgr)
        
        orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        
        axes[row_idx, 0].imshow(orig_rgb)
        axes[row_idx, 0].set_title(f"{cls} - Original Scan\n({Path(sample_p).stem})", fontsize=10, fontweight="bold")
        axes[row_idx, 0].axis("off")
        
        axes[row_idx, 1].imshow(overlay_rgb)
        axes[row_idx, 1].set_title(f"{cls} - Segmented Contour & ROI\n(Green: Parenchyma, Cyan: Box)", fontsize=10, fontweight="bold")
        axes[row_idx, 1].axis("off")
        
        axes[row_idx, 2].imshow(crop_rgb)
        axes[row_idx, 2].set_title(f"{cls} - Cropped ROI (224x224)\n(Aspect Preserved Letterbox)", fontsize=10, fontweight="bold")
        axes[row_idx, 2].axis("off")

    plt.suptitle("Ovarian Ultrasound Parenchymal Segmentation & ROI Extraction Gallery", fontsize=13, fontweight="bold")
    plt.tight_layout()
    
    fig_out = "results/figures/all_segmented_samples.png"
    plt.savefig(fig_out, dpi=200)
    plt.close()
    print(f"[DONE] Saved clinical gallery to: {fig_out}")


if __name__ == "__main__":
    main()
