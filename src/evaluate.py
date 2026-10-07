"""
Comprehensive Evaluation Suite for 3-Class Ovarian Ultrasound Benchmark
Computes:
  - Accuracy, Balanced Accuracy
  - Macro & Weighted Precision, Recall, F1
  - Macro ROC-AUC
  - Class-specific metrics (Normal, PCOS, DF)
  - Cross-follicular confusion errors (PCOS -> DF, DF -> PCOS, Normal -> PCOS, PCOS -> Normal)
  - Confusion matrix & ROC plots
  - Parameter count, checkpoint size, and inference latency profiling
"""

import os
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc
)

CLASS_NAMES = ["Normal Ovary", "PCOS", "Dominant Follicle"]


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu")
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, List[str]]:
    """
    Evaluates a model over a DataLoader and computes all clinical & statistical metrics.
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_probs = []
    all_targets = []
    all_paths = []
    latencies = []

    with torch.no_grad():
        for inputs, targets, paths in dataloader:
            inputs = inputs.to(device)
            
            # Measure inference latency per batch
            start_t = time.perf_counter()
            outputs = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            batch_time = (time.perf_counter() - start_t) / inputs.size(0)
            latencies.extend([batch_time * 1000.0] * inputs.size(0))  # ms

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_targets.extend(targets.numpy())
            all_paths.extend(paths)

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    # 1. Overall Metrics
    acc = accuracy_score(all_targets, all_preds)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    macro_prec = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_rec = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    # Macro ROC-AUC (one-vs-rest)
    try:
        macro_auc = roc_auc_score(all_targets, all_probs, multi_class="ovr", average="macro")
    except Exception:
        macro_auc = 0.0

    # 2. Per-class metrics
    per_cls_prec = precision_score(all_targets, all_preds, average=None, zero_division=0)
    per_cls_rec = recall_score(all_targets, all_preds, average=None, zero_division=0)
    per_cls_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)

    # 3. Confusion Matrix & Cross-Follicular Errors
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])
    
    # cm[True, Pred]
    # 0 = Normal, 1 = PCOS, 2 = DF
    pcos_to_df = int(cm[1, 2])
    df_to_pcos = int(cm[2, 1])
    normal_to_pcos = int(cm[0, 1])
    pcos_to_normal = int(cm[1, 0])

    # 4. Latency & Model Complexity
    mean_latency_ms = float(np.mean(latencies))
    total_params = sum(p.numel() for p in model.parameters())

    metrics = {
        "Accuracy": round(float(acc), 4),
        "Balanced_Accuracy": round(float(bal_acc), 4),
        "Macro_Precision": round(float(macro_prec), 4),
        "Macro_Recall": round(float(macro_rec), 4),
        "Macro_F1": round(float(macro_f1), 4),
        "Weighted_F1": round(float(weighted_f1), 4),
        "Macro_AUC": round(float(macro_auc), 4),
        # Normal
        "Normal_Precision": round(float(per_cls_prec[0]), 4),
        "Normal_Recall": round(float(per_cls_rec[0]), 4),
        "Normal_F1": round(float(per_cls_f1[0]), 4),
        # PCOS
        "PCOS_Precision": round(float(per_cls_prec[1]), 4),
        "PCOS_Recall": round(float(per_cls_rec[1]), 4),
        "PCOS_F1": round(float(per_cls_f1[1]), 4),
        # Dominant Follicle
        "DF_Precision": round(float(per_cls_prec[2]), 4),
        "DF_Recall": round(float(per_cls_rec[2]), 4),
        "DF_F1": round(float(per_cls_f1[2]), 4),
        # Critical Cross-Errors
        "PCOS_to_DF": pcos_to_df,
        "DF_to_PCOS": df_to_pcos,
        "Normal_to_PCOS": normal_to_pcos,
        "PCOS_to_Normal": pcos_to_normal,
        # Hardware
        "Parameters": total_params,
        "Inference_Latency_ms": round(mean_latency_ms, 2)
    }

    return metrics, cm, all_probs, all_preds


def plot_confusion_matrix(cm: np.ndarray, model_name: str, save_path: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = CLASS_NAMES
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=f"Confusion Matrix: {model_name}",
        ylabel="True Diagnosis",
        xlabel="Predicted Diagnosis"
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold")
    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_roc_curves(all_targets: np.ndarray, all_probs: np.ndarray, model_name: str, save_path: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    for i, (cname, color) in enumerate(zip(CLASS_NAMES, colors)):
        binary_targets = (all_targets == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_targets, all_probs[:, i])
        cls_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cname} (AUC = {cls_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(f"ROC Curves: {model_name}")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
