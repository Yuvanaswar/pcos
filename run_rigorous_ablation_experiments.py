"""
Rigorous, Verifiable Ablation Study Execution Script
Conducts controlled empirical experiments with zero simulated data.
Saves raw logs, training history, and evaluation outputs to results/ablation_study/
"""

import os
import sys
import time
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.models.factory import create_model
from src.preprocessing import UltrasoundPreprocessingPipeline, inpaint_calipers, letterbox_image
from src.losses import FollicleAwareFocalLoss
from src.evaluate import evaluate_model, CLASS_NAMES

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class CustomAblationDataset(Dataset):
    def __init__(self, df, split="train", is_train=True, use_inpaint=True, use_letterbox=True, use_aug=True):
        self.data = df[df["split"] == split].reset_index(drop=True)
        self.is_train = is_train
        self.use_inpaint = use_inpaint
        self.use_letterbox = use_letterbox
        self.use_aug = use_aug
        
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row["image_path"]
        target = int(row["class_id"])

        bgr = cv2.imread(img_path)
        if bgr is None:
            raise FileNotFoundError(f"Missing image: {img_path}")
        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 1. Inpainting ablation
        if self.use_inpaint:
            img_rgb = inpaint_calipers(img_rgb)

        # 2. Augmentation ablation
        if self.is_train and self.use_aug:
            if random.random() < 0.5:
                img_rgb = np.ascontiguousarray(np.fliplr(img_rgb))
            angle = random.uniform(-7.0, 7.0)
            h, w = img_rgb.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img_rgb = cv2.warpAffine(img_rgb, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            gain = random.uniform(0.90, 1.10)
            img_rgb = np.clip(img_rgb.astype(np.float32) * gain, 0, 255).astype(np.uint8)

        # 3. Letterbox resize vs Anamorphic Stretch
        if self.use_letterbox:
            resized, _, _ = letterbox_image(img_rgb, target_size=(224, 224))
        else:
            resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

        # 4. Normalize
        arr = resized.astype(np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float()

        return tensor, target, img_path

def train_and_eval_variant(
    variant_id,
    variant_name,
    df,
    device,
    epochs=5,
    batch_size=16,
    lr=3e-4,
    use_sampler=True,
    criterion_type="focal",
    use_inpaint=True,
    use_letterbox=True,
    use_aug=True,
    seed=42,
    log_file=None
):
    def print_log(msg):
        print(msg)
        if log_file:
            log_file.write(msg + "\n")
            log_file.flush()

    set_seed(seed)
    print_log("\n" + "=" * 95)
    print_log(f"RUNNING EXPERIMENT: [{variant_id}] {variant_name}")
    print_log("=" * 95)
    print_log(f"Seed: {seed} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print_log(f"Inpainting: {use_inpaint} | Letterbox: {use_letterbox} | Augmentation: {use_aug} | Sampler: {'Weighted' if use_sampler else 'Uniform'} | Loss: {criterion_type}")

    train_ds = CustomAblationDataset(df, split="train", is_train=True, use_inpaint=use_inpaint, use_letterbox=use_letterbox, use_aug=use_aug)
    val_ds = CustomAblationDataset(df, split="val", is_train=False, use_inpaint=use_inpaint, use_letterbox=use_letterbox, use_aug=False)
    test_ds = CustomAblationDataset(df, split="test", is_train=False, use_inpaint=use_inpaint, use_letterbox=use_letterbox, use_aug=False)

    if use_sampler:
        targets = train_ds.data["class_id"].values
        class_counts = np.bincount(targets)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[targets]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Initialize Model
    model, _ = create_model("ConvNeXt_Tiny_V6", num_classes=3, pretrained=True)
    model.to(device)

    # Loss
    if criterion_type == "focal":
        criterion = FollicleAwareFocalLoss(gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    start_train = time.time()
    best_val_f1 = -1.0
    best_model_state = None

    history = []
    for ep in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        train_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vx, vy, _ in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                val_loss += criterion(model(vx), vy).item() * len(vy)
        val_loss /= len(val_ds)

        val_m, val_cm, _, _ = evaluate_model(model, val_loader, device=device)
        ep_dur = time.time() - ep_t0

        history.append({
            "epoch": ep,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_m["Accuracy"], 4),
            "val_f1": round(val_m["Macro_F1"], 4),
            "pcos_rec": round(val_m["PCOS_Recall"], 4),
            "duration_sec": round(ep_dur, 2)
        })

        print_log(f"Epoch {ep:02d}/{epochs:02d} ({ep_dur:4.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_m['Accuracy']*100:.2f}% | Val Macro-F1: {val_m['Macro_F1']*100:.2f}% | PCOS Rec: {val_m['PCOS_Recall']*100:.2f}%")

        if val_m["Macro_F1"] > best_val_f1:
            best_val_f1 = val_m["Macro_F1"]
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    total_train_time = time.time() - start_train
    print_log(f"Training completed in {total_train_time:.1f}s. Evaluating best checkpoint...")

    # Load best weights
    model.load_state_dict(best_model_state)
    model.to(device).eval()

    # Latency measurement
    dummy = torch.randn(1, 3, 224, 224).to(device)
    for _ in range(5):
        _ = model(dummy)
    latencies = []
    for _ in range(20):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy)
        latencies.append((time.perf_counter() - t0) * 1000)
    avg_latency = float(np.mean(latencies))

    # Final Validation evaluation
    final_val, val_cm, _, _ = evaluate_model(model, val_loader, device=device)
    # Final Test evaluation
    final_test, test_cm, _, _ = evaluate_model(model, test_loader, device=device)

    print_log("-" * 95)
    print_log(f"FINAL VALIDATION: Acc: {final_val['Accuracy']*100:.2f}% | Macro-F1: {final_val['Macro_F1']*100:.2f}% | PCOS Recall: {final_val['PCOS_Recall']*100:.2f}% | PCOS->DF: {final_val['PCOS_to_DF']}")
    print_log(f"FINAL TEST SET : Acc: {final_test['Accuracy']*100:.2f}% | Macro-F1: {final_test['Macro_F1']*100:.2f}% | PCOS Recall: {final_test['PCOS_Recall']*100:.2f}% | PCOS->DF: {final_test['PCOS_to_DF']}")
    print_log(f"Inference Latency: {avg_latency:.2f} ms")

    return {
        "variant_id": variant_id,
        "variant_name": variant_name,
        "seed": seed,
        "train_time_sec": round(total_train_time, 2),
        "latency_ms": round(avg_latency, 2),
        "val_accuracy": round(final_val["Accuracy"], 4),
        "val_balanced_acc": round(final_val["Balanced_Accuracy"], 4),
        "val_macro_f1": round(final_val["Macro_F1"], 4),
        "val_macro_auc": round(final_val["Macro_AUC"], 4),
        "val_pcos_recall": round(final_val["PCOS_Recall"], 4),
        "val_pcos_f1": round(final_val["PCOS_F1"], 4),
        "val_pcos_to_df": int(final_val["PCOS_to_DF"]),
        "test_accuracy": round(final_test["Accuracy"], 4),
        "test_macro_f1": round(final_test["Macro_F1"], 4),
        "test_pcos_recall": round(final_test["PCOS_Recall"], 4),
        "test_pcos_to_df": int(final_test["PCOS_to_DF"]),
        "history": history
    }

def main():
    out_dir = Path("results/ablation_study")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "ablation_experiments_raw.log"
    log_file = open(log_path, "w", encoding="utf-8")

    manifest = "manifest.csv" if os.path.exists("manifest.csv") else "results/manifest.csv"
    df = pd.read_csv(manifest)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_file.write(f"Hardware Compute Device: {device}\n")
    log_file.write(f"PyTorch Version: {torch.__version__}\n")
    log_file.write(f"Dataset Manifest: {manifest} ({len(df)} total scans)\n")

    # Define 6 Systematic Controlled Ablation Variants
    variants = [
        {
            "id": "VAR-1",
            "name": "Naive Baseline (CE Loss, Uniform Sampling, No Aug)",
            "use_sampler": False,
            "criterion_type": "ce",
            "use_inpaint": True,
            "use_letterbox": True,
            "use_aug": False
        },
        {
            "id": "VAR-2",
            "name": "Add Data Augmentation (Flip, Rotation, Gain)",
            "use_sampler": False,
            "criterion_type": "ce",
            "use_inpaint": True,
            "use_letterbox": True,
            "use_aug": True
        },
        {
            "id": "VAR-3",
            "name": "Add Class Balancing (WeightedRandomSampler)",
            "use_sampler": True,
            "criterion_type": "ce",
            "use_inpaint": True,
            "use_letterbox": True,
            "use_aug": True
        },
        {
            "id": "VAR-4",
            "name": "Proposed System (FollicleAwareFocalLoss + Weighted + Aug)",
            "use_sampler": True,
            "criterion_type": "focal",
            "use_inpaint": True,
            "use_letterbox": True,
            "use_aug": True
        },
        {
            "id": "VAR-5",
            "name": "Ablate Caliper Inpainting (Raw Scans with Machine Artifacts)",
            "use_sampler": True,
            "criterion_type": "focal",
            "use_inpaint": False,
            "use_letterbox": True,
            "use_aug": True
        },
        {
            "id": "VAR-6",
            "name": "Ablate Letterboxing (Anamorphic Direct Bilinear Stretch)",
            "use_sampler": True,
            "criterion_type": "focal",
            "use_inpaint": True,
            "use_letterbox": False,
            "use_aug": True
        }
    ]

    all_results = []
    for v in variants:
        res = train_and_eval_variant(
            variant_id=v["id"],
            variant_name=v["name"],
            df=df,
            device=device,
            epochs=3,  # 3 epochs for fast, clean, reproducible convergence comparison
            batch_size=16,
            lr=3e-4,
            use_sampler=v["use_sampler"],
            criterion_type=v["criterion_type"],
            use_inpaint=v["use_inpaint"],
            use_letterbox=v["use_letterbox"],
            use_aug=v["use_aug"],
            seed=42,
            log_file=log_file
        )
        all_results.append(res)

    # Save summary table
    summary_df = pd.DataFrame([{
        "Variant_ID": r["variant_id"],
        "Variant_Name": r["variant_name"],
        "Training_Time_s": r["train_time_sec"],
        "Latency_ms": r["latency_ms"],
        "Val_Acc_%": round(r["val_accuracy"] * 100, 2),
        "Val_Macro_F1_%": round(r["val_macro_f1"] * 100, 2),
        "Val_PCOS_Recall_%": round(r["val_pcos_recall"] * 100, 2),
        "Val_PCOS_to_DF_Errors": r["val_pcos_to_df"],
        "Test_Acc_%": round(r["test_accuracy"] * 100, 2),
        "Test_Macro_F1_%": round(r["test_macro_f1"] * 100, 2),
        "Test_PCOS_Recall_%": round(r["test_pcos_recall"] * 100, 2),
        "Test_PCOS_to_DF_Errors": r["test_pcos_to_df"]
    } for r in all_results])

    csv_out = out_dir / "ablation_results_summary.csv"
    summary_df.to_csv(csv_out, index=False)

    json_out = out_dir / "ablation_results_full.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 95)
    print("MASTER CONTROLLED ABLATION RESULTS SUMMARY")
    print("=" * 95)
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary CSV to: {csv_out}")
    print(f"Saved detailed JSON to: {json_out}")
    print(f"Saved raw execution log to: {log_path}")

    log_file.close()

if __name__ == "__main__":
    main()
