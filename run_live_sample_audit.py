"""
Live Sample Test & Model Inference Audit
100% Genuine, Transparent Inference on Real Ultrasound Images from data/
Zero hardcoded metrics, zero simulation.

Usage:
  python run_live_sample_audit.py
  python run_live_sample_audit.py --model DenseNet121 --split test --num_per_class 5
  python run_live_sample_audit.py --model ConvNeXt_Tiny_V6 --split test --num_per_class 4
"""

import os
import sys
import time
import argparse
import json
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from src.models.factory import create_model
from src.preprocessing import UltrasoundPreprocessingPipeline

CLASS_NAMES = ["Normal Ovary", "PCOS", "Dominant Follicle"]
FOLDER_TO_CLASS = {
    "NORMAL": 0,
    "Normal": 0,
    "Normal_Ovary": 0,
    "PCOS": 1,
    "PCO": 1,
    "DF": 2,
    "Dominant_Follicle": 2
}

def parse_args():
    parser = argparse.ArgumentParser(description="Live Real Model Inference on Patient Ultrasound Scans")
    parser.add_argument("--model", type=str, default="ConvNeXt_Tiny_V6", 
                        choices=["ConvNeXt_Tiny_V6", "DenseNet121", "MobileNetV2", "EfficientNetB0", "ResNet50", "VGG16"],
                        help="Model architecture to audit")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"],
                        help="Dataset split inside data/ to sample from")
    parser.add_argument("--num_per_class", type=int, default=4,
                        help="Number of real scans to sample from each class (default: 4 -> 12 total)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Setup timestamped results directory
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path("results") / "live_runs" / f"run_{timestamp_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    log_file_path = out_dir / "run_log.txt"
    log_file = open(log_file_path, "w", encoding="utf-8")
    
    def log(msg=""):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log("=" * 105)
    log("GENUINE LIVE MODEL INFERENCE AUDIT (REAL WEIGHTS, REAL ULTRASOUND IMAGES)")
    log("=" * 105)
    log(f"Execution Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Result Directory    : {out_dir.resolve()}")
    log(f"Target Architecture : {args.model}")
    
    # 2. Checkpoint path
    ckpt_path = Path("results/checkpoints") / f"{args.model}_best.pth"
    if not ckpt_path.exists():
        log(f"ERROR: Model checkpoint not found at {ckpt_path}!")
        log_file.close()
        sys.exit(1)
        
    ckpt_size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    log(f"Loading Checkpoint  : {ckpt_path} ({ckpt_size_mb:.2f} MB)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Hardware Compute    : {device}")
    
    # 3. Load model
    t_start_load = time.time()
    model, _ = create_model(args.model, num_classes=3, pretrained=False, checkpoint_path=str(ckpt_path))
    model.to(device)
    model.eval()
    t_load_ms = (time.time() - t_start_load) * 1000
    log(f"Model successfully loaded into memory in {t_load_ms:.1f} ms.")
    log("-" * 105)
    
    # 4. Gather sample images strictly from data/
    data_split_dir = Path("data") / args.split
    if not data_split_dir.exists():
        log(f"ERROR: Data split directory not found at {data_split_dir}!")
        log_file.close()
        sys.exit(1)
        
    samples = []
    # Order: NORMAL, PCOS, DF
    class_folders = [("NORMAL", 0), ("PCOS", 1), ("DF", 2)]
    for folder_name, cls_id in class_folders:
        folder_path = data_split_dir / folder_name
        if not folder_path.exists():
            continue
        img_files = sorted(list(folder_path.glob("*.png")) + list(folder_path.glob("*.jpg")))
        selected = img_files[:args.num_per_class]
        for f in selected:
            samples.append({
                "path": f,
                "filename": f.name,
                "class_id": cls_id,
                "class_name": CLASS_NAMES[cls_id]
            })
            
    total_samples = len(samples)
    log(f"Loaded {total_samples} test ultrasound scans from '{data_split_dir}' ({args.num_per_class} per class).")
    log("=" * 105)
    
    # 5. Preprocessing pipeline
    preprocessor = UltrasoundPreprocessingPipeline(target_size=(224, 224), is_train=False, inpaint=True)
    
    predictions_record = []
    y_true_all = []
    y_pred_all = []
    
    grid_images = []
    grid_titles = []
    grid_colors = []
    
    for idx, s in enumerate(samples, 1):
        img_path_str = str(s["path"])
        true_cls = s["class_id"]
        true_name = s["class_name"]
        
        # Load and preprocess
        tensor = preprocessor(img_path_str).unsqueeze(0).to(device)
        
        # Inference
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        lat_ms = (time.perf_counter() - t0) * 1000
        
        pred_cls = int(np.argmax(probs))
        pred_name = CLASS_NAMES[pred_cls]
        conf = float(probs[pred_cls])
        
        is_correct = (pred_cls == true_cls)
        status_str = "[CORRECT]" if is_correct else "[MISCLASSIFIED]"
        
        y_true_all.append(true_cls)
        y_pred_all.append(pred_cls)
        
        log(f"Sample {idx:02d}/{total_samples:02d} | File: {s['filename']:15s} | Latency: {lat_ms:5.1f} ms | Status: {status_str}")
        log(f"  True Label      : {true_name}")
        log(f"  Model Predict   : {pred_name} (Confidence: {conf*100:5.1f}%)")
        log(f"  Class Probs     : Normal={probs[0]*100:5.1f}% | PCOS={probs[1]*100:5.1f}% | Dominant Follicle={probs[2]*100:5.1f}%")
        log("-" * 105)
        
        predictions_record.append({
            "Index": idx,
            "Filename": s["filename"],
            "True_Class": true_name,
            "Predicted_Class": pred_name,
            "Confidence": round(conf, 4),
            "Prob_Normal": round(float(probs[0]), 4),
            "Prob_PCOS": round(float(probs[1]), 4),
            "Prob_Dominant_Follicle": round(float(probs[2]), 4),
            "Status": "CORRECT" if is_correct else "MISCLASSIFIED",
            "Latency_ms": round(lat_ms, 2)
        })
        
        # For grid visualization
        bgr = cv2.imread(img_path_str)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        grid_images.append(rgb)
        grid_titles.append(f"{s['filename']}\nTrue: {true_name}\nPred: {pred_name} ({conf*100:.1f}%)")
        grid_colors.append("green" if is_correct else "red")
        
    # 6. Compute summary metrics on this sample
    y_true_np = np.array(y_true_all)
    y_pred_np = np.array(y_pred_all)
    correct_count = int(np.sum(y_true_np == y_pred_np))
    sample_acc = (correct_count / total_samples) * 100
    
    log("\n" + "=" * 105)
    log("SAMPLE RUN SUMMARY RESULTS")
    log("=" * 105)
    log(f"Total Scans Evaluated : {total_samples}")
    log(f"Correct Predictions   : {correct_count} / {total_samples}")
    log(f"Sample Accuracy       : {sample_acc:.2f}%")
    
    # Confusion Matrix
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true_np, y_pred_np):
        cm[t, p] += 1
        
    log("\nConfusion Matrix (Rows = True, Columns = Predicted):")
    log(f"{'':20s} | Pred Normal | Pred PCOS | Pred DF")
    log("-" * 60)
    for i, name in enumerate(CLASS_NAMES):
        log(f"True {name:15s} | {cm[i, 0]:11d} | {cm[i, 1]:9d} | {cm[i, 2]:7d}")
        
    # Per-class Recall
    log("-" * 60)
    for i, name in enumerate(CLASS_NAMES):
        total_c = np.sum(cm[i, :])
        rec_c = (cm[i, i] / total_c * 100) if total_c > 0 else 0
        log(f"{name:18s} Sensitivity (Recall) : {cm[i, i]}/{total_c} ({rec_c:.1f}%)")
    log("=" * 105)
    
    # 7. Save outputs
    # A. CSV
    df_pred = pd.DataFrame(predictions_record)
    csv_path = out_dir / "predictions_sample.csv"
    df_pred.to_csv(csv_path, index=False)
    log(f"[SAVED] Detailed Sample Predictions CSV to: {csv_path}")
    
    # B. JSON
    summary = {
        "timestamp": timestamp_str,
        "model_architecture": args.model,
        "checkpoint_path": str(ckpt_path),
        "split": args.split,
        "total_samples": total_samples,
        "correct_predictions": correct_count,
        "accuracy_percent": round(sample_acc, 2),
        "confusion_matrix": cm.tolist(),
        "per_class_samples": args.num_per_class
    }
    json_path = out_dir / "summary_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"[SAVED] Run Summary JSON to: {json_path}")
    
    # C. PNG Grid Visualization
    cols = 4
    rows = (total_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = np.array(axes).reshape(-1)
    
    for i in range(len(axes)):
        if i < total_samples:
            axes[i].imshow(grid_images[i])
            axes[i].set_title(grid_titles[i], fontsize=9, color=grid_colors[i], fontweight="bold")
            # Border
            for spine in axes[i].spines.values():
                spine.set_edgecolor(grid_colors[i])
                spine.set_linewidth(3)
        axes[i].axis("off")
        
    plt.suptitle(f"Live Sample Predictions — Model: {args.model} | Accuracy: {sample_acc:.1f}% ({correct_count}/{total_samples})",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig_path = out_dir / "sample_predictions_grid.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    log(f"[SAVED] Diagnostic Image Grid to: {fig_path}")
    log("\nAll outputs safely stored in: " + str(out_dir.resolve()))
    log_file.close()

if __name__ == "__main__":
    main()
