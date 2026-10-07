"""
Script to construct and execute notebooks/02_IMAGE_AUDIT.ipynb
Fulfills all requirements of Part B - Image Quality Audit.
"""

import os
import sys
from pathlib import Path
from src.nb_utils import NotebookBuilder

def build_02_notebook():
    nb = NotebookBuilder(title="02_IMAGE_AUDIT: Ultrasound Quality & Artifact Characterization")
    
    # Cell 1: Markdown
    nb.add_markdown(r"""# 02. IMAGE QUALITY AUDIT: Ovarian Ultrasound Characteristics
## Resolution, Aspect Ratio, Intensity Distributions, and Artifact Screening

### Clinical Audit Objectives:
1. Analyze resolution, aspect ratios, and intensity distributions across all three classes (**Normal Ovary**, **PCOS**, **Dominant Follicle**).
2. Screen for common ultrasound artifacts:
   - Calipers and electronic measurement crosses
   - Text overlays and machine telemetry
   - Heavy acoustic shadowing and black border padding
   - Dynamic range and contrast anomalies
3. **Safety Protocol:** Do NOT automatically delete any images. Flag and categorize them transparently.
4. Export quantitative results to `results/image_quality_audit.csv`.
""")

    # Cell 2: Code - Imports & Manifest Loading
    nb.add_code(r"""# Cell 1: Imports and Manifest Verification
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import skew

# Load manifest
manifest_path = "manifest.csv" if os.path.exists("manifest.csv") else "../manifest.csv"
if not os.path.exists(manifest_path):
    manifest_path = "results/manifest.csv"

df = pd.read_csv(manifest_path)
print(f"[Manifest Loaded] Total images to audit: {len(df)}")
print(df["class_name"].value_counts())
""")

    # Cell 3: Code - Class Representative Analysis & Histograms
    nb.add_code(r"""# Cell 2: Per-Class Representative Analysis, Intensity Stats & Histograms
classes = [("Normal Ovary", 0), ("PCOS", 1), ("Dominant Follicle", 2)]

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for row_idx, (cname, cid) in enumerate(classes):
    sub = df[df["class_id"] == cid]
    sample_row = sub.iloc[0]
    img_path = sample_row["image_path"]
    
    # Read grayscale image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {img_path}")
        
    h, w = img.shape
    aspect_ratio = round(w / h, 3)
    mean_val = np.mean(img)
    std_val = np.std(img)
    median_val = np.median(img)
    min_val, max_val = np.min(img), np.max(img)
    skew_val = skew(img.ravel())
    
    # Subplot 1: Image
    axes[row_idx, 0].imshow(img, cmap="gray")
    axes[row_idx, 0].set_title(f"{cname}\nRes: {w}x{h} | AR: {aspect_ratio}", fontsize=11, fontweight="bold")
    axes[row_idx, 0].axis("off")
    
    # Subplot 2: Intensity Histogram
    axes[row_idx, 1].hist(img.ravel(), bins=64, range=(0, 256), color="#377eb8", edgecolor="black", alpha=0.7)
    axes[row_idx, 1].set_title(f"{cname} Histogram\nMean: {mean_val:.1f} | Std: {std_val:.1f}", fontsize=10)
    axes[row_idx, 1].set_xlabel("Pixel Intensity (0-255)")
    axes[row_idx, 1].set_ylabel("Frequency")
    axes[row_idx, 1].grid(True, linestyle="--", alpha=0.5)
    
    # Subplot 3: Stats Text Summary
    stats_text = (
        f"Class: {cname}\n"
        f"File: {sample_row['patient_id']}\n"
        f"-------------------------------\n"
        f"Resolution:     {w} x {h}\n"
        f"Aspect Ratio:   {aspect_ratio:.3f}\n"
        f"Min Intensity:  {min_val}\n"
        f"Max Intensity:  {max_val}\n"
        f"Mean Intensity: {mean_val:.2f}\n"
        f"Std Dev:        {std_val:.2f}\n"
        f"Median:         {median_val:.1f}\n"
        f"Skewness:       {skew_val:.2f}\n"
    )
    axes[row_idx, 2].text(0.1, 0.5, stats_text, fontsize=11, family="monospace", va="center")
    axes[row_idx, 2].axis("off")

plt.tight_layout()
os.makedirs("results/figures", exist_ok=True)
plt.savefig("results/figures/image_audit_histograms.png", dpi=200)
plt.close()
print("✓ Saved representative histogram analysis to: results/figures/image_audit_histograms.png")
""")

    # Cell 4: Code - Artifact Detection & Quality Scoring across ALL 572 Scans
    nb.add_code(r"""# Cell 3: Comprehensive Artifact Screening & Metric Calculation
audit_rows = []

print(f"Auditing all {len(df)} scans for resolution, aspect ratio, calipers, borders, and text overlays...")

for idx, row in df.iterrows():
    img_path = row["image_path"]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
        
    h, w = img.shape
    ar = round(w / h, 4)
    
    # 1. Intensity stats
    mean_val = float(np.mean(img))
    std_val = float(np.std(img))
    min_val = int(np.min(img))
    max_val = int(np.max(img))
    dynamic_range = max_val - min_val
    
    # 2. Black border padding estimation (percentage of perimeter within 10% of image with intensity < 5)
    border_mask = np.zeros_like(img, dtype=bool)
    pad_h, pad_w = int(h * 0.08), int(w * 0.08)
    border_mask[:pad_h, :] = True
    border_mask[-pad_h:, :] = True
    border_mask[:, :pad_w] = True
    border_mask[:, -pad_w:] = True
    border_pixels = img[border_mask]
    black_border_pct = float(np.mean(border_pixels < 10) * 100)
    
    # 3. High-contrast marker / caliper detection
    # Electronic calipers/markers are typically isolated ultra-white (intensity >= 250) structures
    bright_mask = (img >= 250).astype(np.uint8)
    # Detect small connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bright_mask)
    caliper_candidates = 0
    text_candidates = 0
    for s_idx in range(1, num_labels):
        area = stats[s_idx, cv2.CC_STAT_AREA]
        bw = stats[s_idx, cv2.CC_STAT_WIDTH]
        bh = stats[s_idx, cv2.CC_STAT_HEIGHT]
        # Crosses/calipers typically 5 to 50 pixels in area with aspect ratio close to 1
        if 6 <= area <= 60 and 0.5 <= (bw / max(bh, 1)) <= 2.0:
            caliper_candidates += 1
        # Text characters usually clustered in header/footer with height 8-20
        elif 15 <= area <= 200:
            text_candidates += 1

    # 4. Poor-quality / Low-contrast flag
    is_low_contrast = (std_val < 25.0) or (dynamic_range < 120)
    is_excessive_border = black_border_pct > 75.0
    has_calipers = caliper_candidates >= 2
    has_text = text_candidates >= 5
    
    flags = []
    if is_low_contrast:
        flags.append("LowContrast")
    if is_excessive_border:
        flags.append("HighPadding")
    if has_calipers:
        flags.append("CalipersDetected")
    if has_text:
        flags.append("TextOverlay")
        
    quality_flag = ";".join(flags) if flags else "Clean"
    
    audit_rows.append({
        "image_path": img_path,
        "class_name": row["class_name"],
        "class_id": row["class_id"],
        "split": row["split"],
        "patient_id": row["patient_id"],
        "width": w,
        "height": h,
        "aspect_ratio": ar,
        "mean_intensity": round(mean_val, 2),
        "std_intensity": round(std_val, 2),
        "min_intensity": min_val,
        "max_intensity": max_val,
        "dynamic_range": dynamic_range,
        "black_border_pct": round(black_border_pct, 2),
        "caliper_count": caliper_candidates,
        "text_count": text_candidates,
        "quality_flag": quality_flag
    })

audit_df = pd.DataFrame(audit_rows)
print(f"Audit completed for all {len(audit_df)} images.")
""")

    # Cell 5: Code - Export CSV and Summaries
    nb.add_code(r"""# Cell 4: Export CSV & Quality Flag Summary
os.makedirs("results", exist_ok=True)
audit_csv_path = "results/image_quality_audit.csv"
audit_df.to_csv(audit_csv_path, index=False)
print(f"✓ Saved image quality audit to: {audit_csv_path}")

print("\n" + "=" * 70)
print("IMAGE QUALITY AUDIT SUMMARY")
print("=" * 70)
print(f"Total Scans Audited:             {len(audit_df)}")
print(f"Aspect Ratio Range:              {audit_df['aspect_ratio'].min():.3f} - {audit_df['aspect_ratio'].max():.3f}")
print(f"Mean Image Dimensions:           {int(audit_df['width'].mean())} x {int(audit_df['height'].mean())}")
print(f"Clean Scans (No Major Artifacts): {(audit_df['quality_flag'] == 'Clean').sum()} ({((audit_df['quality_flag'] == 'Clean').sum()/len(audit_df))*100:.1f}%)")

print("\nArtifact Breakdown across Dataset:")
for flag in ["CalipersDetected", "TextOverlay", "HighPadding", "LowContrast"]:
    cnt = audit_df["quality_flag"].str.contains(flag).sum()
    print(f"  - {flag:20s}: {cnt} scans ({cnt/len(audit_df)*100:.1f}%)")

print("\nAudit Head (5 rows):")
print(audit_df[["patient_id", "class_name", "width", "height", "aspect_ratio", "black_border_pct", "quality_flag"]].head())
""")

    # Cell 6: Code - Artifact Visualization
    nb.add_code(r"""# Cell 5: Artifact Examples Visualization
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

flag_types = ["CalipersDetected", "TextOverlay", "HighPadding", "Clean"]
titles = [
    "Caliper / Measurement Marker",
    "Text Overlay / Annotations",
    "Acoustic Border Padding",
    "Clean Ovarian Parenchyma"
]

for idx, (ftype, title) in enumerate(zip(flag_types, titles)):
    if ftype == "Clean":
        match_rows = audit_df[audit_df["quality_flag"] == "Clean"]
    else:
        match_rows = audit_df[audit_df["quality_flag"].str.contains(ftype)]
        
    if len(match_rows) > 0:
        sample_path = match_rows.iloc[0]["image_path"]
        img = cv2.imread(sample_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(img_rgb)
        axes[idx].set_title(f"{title}\n({match_rows.iloc[0]['patient_id']})", fontsize=10, fontweight="bold")
    else:
        axes[idx].text(0.5, 0.5, "None Found", ha="center", va="center")
    axes[idx].axis("off")

plt.tight_layout()
plt.savefig("results/figures/artifact_examples.png", dpi=200)
plt.close()
print("✓ Saved artifact examples to: results/figures/artifact_examples.png")
""")

    out_path = "notebooks/02_IMAGE_AUDIT.ipynb"
    nb.execute_and_save(out_path, working_dir=".")
    print(f"Successfully generated and executed {out_path}!")

if __name__ == "__main__":
    build_02_notebook()
