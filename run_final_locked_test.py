"""
Part O & Part P: Single-Pass Locked Test Set Evaluation and Final Master Table.
Evaluates:
  1. Best Baseline CNN: DenseNet-121
  2. Baseline CNN Baseline: ResNet-50
  3. Legacy Deep Baseline: VGG-16
  4. Lightweight Baseline: MobileNet-V2
  5. Current Model: ConvNeXt-Tiny V6
  6. Best Transformer: Swin-Tiny
  7. Best ROI Pipeline: U-Net ROI + ConvNeXt-Tiny V6
Produces:
  FINAL_COMPARISON.csv
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.factory import create_model
from src.dataset import OvarianUltrasoundDataset
from src.segmentation.roi_extractor import ROIEnhancedDataset
from src.evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curves, CLASS_NAMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware compute device for Locked Test: {device}")

manifest_path = "manifest.csv" if os.path.exists("manifest.csv") else "results/manifest.csv"
test_base_dataset = OvarianUltrasoundDataset(manifest_path, split="test", is_train=False)
test_loader_direct = DataLoader(test_base_dataset, batch_size=16, shuffle=False)

print(f"Locked Test Set Loaded: {len(test_base_dataset)} ultrasound scans.")

# Define the models and pipelines for final locked comparison
EVAL_CONFIGS = [
    {
        "Model": "DenseNet-121",
        "Pipeline": "Direct Image (224x224)",
        "Model_Key": "DenseNet121",
        "Checkpoint": "results/checkpoints/DenseNet121_best.pth",
        "Use_ROI": None
    },
    {
        "Model": "ResNet-50",
        "Pipeline": "Direct Image (224x224)",
        "Model_Key": "ResNet50",
        "Checkpoint": "results/checkpoints/ResNet50_best.pth",
        "Use_ROI": None
    },
    {
        "Model": "VGG-16",
        "Pipeline": "Direct Image (224x224)",
        "Model_Key": "VGG16",
        "Checkpoint": "results/checkpoints/VGG16_best.pth",
        "Use_ROI": None
    },
    {
        "Model": "MobileNet-V2",
        "Pipeline": "Direct Image (224x224)",
        "Model_Key": "MobileNetV2",
        "Checkpoint": "results/checkpoints/MobileNetV2_best.pth",
        "Use_ROI": None
    },
    {
        "Model": "ConvNeXt-Tiny V6",
        "Pipeline": "Direct Image (224x224)",
        "Model_Key": "ConvNeXt_Tiny_V6",
        "Checkpoint": "results/checkpoints/ConvNeXt_Tiny_V6_best.pth",
        "Use_ROI": None
    },
    {
        "Model": "Swin-Tiny",
        "Pipeline": "Direct Image (224x224)",
        "Model_Key": "Swin_Tiny",
        "Checkpoint": "results/checkpoints/Swin_Tiny_best.pth",
        "Use_ROI": None
    },
    {
        "Model": "ConvNeXt-Tiny V6",
        "Pipeline": "U-Net ROI Crop (224x224)",
        "Model_Key": "ConvNeXt_Tiny_V6",
        "Checkpoint": "results/checkpoints/ConvNeXt_Tiny_V6_best.pth",
        "Use_ROI": "unet"
    }
]

final_rows = []
test_output_dir = Path("results/test")
test_output_dir.mkdir(parents=True, exist_ok=True)

for cfg in EVAL_CONFIGS:
    m_name = cfg["Model"]
    pipe = cfg["Pipeline"]
    m_key = cfg["Model_Key"]
    ckpt = cfg["Checkpoint"]
    roi_method = cfg["Use_ROI"]
    
    print(f"\nEvaluating Locked Test: {m_name} | {pipe}")
    
    # Load Model
    model, _ = create_model(m_key, num_classes=3, pretrained=True, checkpoint_path=ckpt)
    model.to(device)
    model.eval()
    
    # Choose loader
    if roi_method is None:
        loader = test_loader_direct
    else:
        roi_dataset = ROIEnhancedDataset(test_base_dataset, method=roi_method)
        loader = DataLoader(roi_dataset, batch_size=16, shuffle=False)
        
    metrics, cm, probs, preds = evaluate_model(model, loader, device=device)
    
    # Save test confusion matrix
    cm_path = test_output_dir / f"cm_{m_key}_{'roi' if roi_method else 'direct'}.png"
    plot_confusion_matrix(cm, f"{m_name} ({pipe})", str(cm_path))
    
    row = {
        "Model": m_name,
        "Pipeline": pipe,
        "Accuracy": metrics["Accuracy"],
        "Balanced Accuracy": metrics["Balanced_Accuracy"],
        "Macro Precision": metrics["Macro_Precision"],
        "Macro Recall": metrics["Macro_Recall"],
        "Macro F1": metrics["Macro_F1"],
        "Macro AUC": metrics["Macro_AUC"],
        # Normal
        "Normal Precision": metrics["Normal_Precision"],
        "Normal Recall": metrics["Normal_Recall"],
        "Normal F1": metrics["Normal_F1"],
        # PCOS
        "PCOS Precision": metrics["PCOS_Precision"],
        "PCOS Recall": metrics["PCOS_Recall"],
        "PCOS F1": metrics["PCOS_F1"],
        # Dominant Follicle
        "DF Precision": metrics["DF_Precision"],
        "DF Recall": metrics["DF_Recall"],
        "DF F1": metrics["DF_F1"],
        # Cross Errors
        "PCOS_to_DF": metrics["PCOS_to_DF"],
        "DF_to_PCOS": metrics["DF_to_PCOS"],
        # Hardware
        "Parameters": metrics["Parameters"],
        "Inference Latency": f"{metrics['Inference_Latency_ms']:.1f} ms"
    }
    final_rows.append(row)

final_df = pd.DataFrame(final_rows)

# Sort by Macro F1 descending
final_df = final_df.sort_values(by="Macro F1", ascending=False).reset_index(drop=True)

# Save FINAL_COMPARISON.csv
final_csv_root = "FINAL_COMPARISON.csv"
final_csv_results = "results/FINAL_COMPARISON.csv"

final_df.to_csv(final_csv_root, index=False)
final_df.to_csv(final_csv_results, index=False)

print("\n" + "=" * 110)
print("FINAL LOCKED-TEST MASTER EVALUATION TABLE")
print("=" * 110)
print(final_df.to_string(index=False))

print(f"\n[OK] Saved FINAL_COMPARISON.csv to {final_csv_root} and {final_csv_results}")
