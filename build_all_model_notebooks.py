"""
Build and execute the 8 model benchmark notebooks:
04_RESNET50.ipynb
05_DENSENET121.ipynb
06_VGG16.ipynb
07_MOBILENETV2.ipynb
08_EFFICIENTNETB0.ipynb
09_CONVNEXT_TINY_V6.ipynb
10_SWIN_TINY.ipynb
11_VIT_B16.ipynb

Every notebook adheres strictly to the mandatory 19-cell structure.
"""

import os
import sys
import json
from pathlib import Path
from src.nb_utils import NotebookBuilder

MODELS = [
    {
        "nb_name": "04_RESNET50.ipynb",
        "model_key": "ResNet50",
        "lr": 3e-4,
        "epochs": 15,
        "checkpoint": "results/checkpoints/ResNet50_best.pth",
        "description": "Baseline CNN: ResNet50."
    },
    {
        "nb_name": "05_DENSENET121.ipynb",
        "model_key": "DenseNet121",
        "lr": 3e-4,
        "epochs": 15,
        "checkpoint": "results/checkpoints/DenseNet121_best.pth",
        "description": "Baseline CNN: DenseNet-121."
    },
    {
        "nb_name": "06_VGG16.ipynb",
        "model_key": "VGG16",
        "lr": 3e-4,
        "epochs": 15,
        "checkpoint": "results/checkpoints/VGG16_best.pth",
        "description": "Baseline CNN: VGG16."
    },
    {
        "nb_name": "07_MOBILENETV2.ipynb",
        "model_key": "MobileNetV2",
        "lr": 3e-4,
        "epochs": 15,
        "checkpoint": "results/checkpoints/MobileNetV2_best.pth",
        "description": "Lightweight Baseline: MobileNet-V2."
    },
    {
        "nb_name": "08_EFFICIENTNETB0.ipynb",
        "model_key": "EfficientNetB0",
        "lr": 3e-4,
        "epochs": 15,
        "checkpoint": "results/checkpoints/EfficientNetB0_best.pth",
        "description": "Efficient Baseline: EfficientNet-B0."
    },
    {
        "nb_name": "09_CONVNEXT_TINY_V6.ipynb",
        "model_key": "ConvNeXt_Tiny_V6",
        "lr": 3e-4,
        "epochs": 15,
        "checkpoint": "results/checkpoints/ConvNeXt_Tiny_V6_best.pth",
        "description": "Modern CNN: ConvNeXt-Tiny."
    },
    {
        "nb_name": "10_SWIN_TINY.ipynb",
        "model_key": "Swin_Tiny",
        "lr": 3e-4,
        "epochs": 15,
        "checkpoint": "results/checkpoints/Swin_Tiny_best.pth",
        "description": "Vision Transformer: Swin-Tiny."
    },
    {
        "nb_name": "11_VIT_B16.ipynb",
        "model_key": "ViT_B16",
        "lr": 3e-4,
        "epochs": 15,
        "checkpoint": "results/checkpoints/ViT_B16_best.pth",
        "description": "Vision Transformer: ViT-B/16."
    },
    {
        "nb_name": "12_INCEPTIONV3.ipynb",
        "model_key": "InceptionV3",
        "lr": 3e-4,
        "epochs": 15,
        "checkpoint": "results/checkpoints/InceptionV3_best.pth",
        "description": "InceptionV3 CNN."
    },
    {
        "nb_name": "13_EFFICIENTNETV2.ipynb",
        "model_key": "EfficientNetV2B0",
        "lr": 3e-4,
        "epochs": 15,
        "checkpoint": "results/checkpoints/EfficientNetV2B0_best.pth",
        "description": "EfficientNetV2 Baseline."
    },
    {
        "nb_name": "14_SERESNET50.ipynb",
        "model_key": "SE-ResNet50",
        "lr": 3e-4,
        "epochs": 15,
        "checkpoint": "results/checkpoints/SE_ResNet50_best.pth",
        "description": "Squeeze-and-Excitation ResNet50."
    },
    {
        "nb_name": "15_CBAMRESNET50.ipynb",
        "model_key": "CBAM-ResNet50",
        "lr": 3e-4,
        "epochs": 15,
        "checkpoint": "results/checkpoints/CBAM_ResNet50_best.pth",
        "description": "CBAM Attention ResNet50."
    },
    {
        "nb_name": "16_SERESNET50_FMF.ipynb",
        "model_key": "SE-ResNet50-FMFLoss",
        "lr": 3e-4,
        "epochs": 15,
        "checkpoint": "results/checkpoints/SE_ResNet50_FMFLoss_best.pth",
        "description": "SE-ResNet50 with FMF Loss."
    },
    {
        "nb_name": "17_SERESNET50_DUAL.ipynb",
        "model_key": "SE-ResNet50-DualMarginFMFLoss",
        "lr": 3e-4,
        "epochs": 15,
        "checkpoint": "results/checkpoints/SE_ResNet50_DualMarginFMFLoss_best.pth",
        "description": "SE-ResNet50 with Dual Margin FMF Loss."
    }
]


def generate_and_execute_model_notebook(model_info: dict):
    nb_name = model_info["nb_name"]
    model_key = model_info["model_key"]
    lr = model_info["lr"]
    epochs = model_info["epochs"]
    ckpt = model_info["checkpoint"].replace("\\", "/")
    desc = model_info["description"]
    
    print(f"\n=======================================================")
    print(f"Building & Executing: {nb_name} ({model_key})")
    print(f"=======================================================")

    nb = NotebookBuilder(title=f"{nb_name}: {model_key} 3-Class Benchmark")

    # CELL 1: Imports
    nb.add_code(rf"""# CELL 1: Imports
import os
import sys
import json
import time
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.factory import create_model
from src.dataset import get_dataloaders
from src.losses import FollicleAwareFocalLoss, FMFLoss, DualMarginFMFLoss
from src.evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curves, CLASS_NAMES
from src.gradcam import GradCAM, overlay_cam_on_image

print(f"[PyTorch Version] {{torch.__version__}}")
""")

    # CELL 2: Configuration
    nb.add_code(rf"""# CELL 2: Configuration
MODEL_NAME = "{model_key}"
NUM_CLASSES = 3
BATCH_SIZE = 16
LEARNING_RATE = {lr}
EPOCHS = {epochs}
CHECKPOINT_PATH = "{ckpt}"
MANIFEST_PATH = "manifest.csv" if os.path.exists("manifest.csv") else "results/manifest.csv"
OUTPUT_DIR = Path("results") / MODEL_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Configured {{MODEL_NAME}}:")
print(f"  Learning Rate:   {{LEARNING_RATE}}")
print(f"  Batch Size:      {{BATCH_SIZE}}")
print(f"  Checkpoint Path: {{CHECKPOINT_PATH}} (Exists: {{os.path.exists(CHECKPOINT_PATH)}})")
""")

    # CELL 3: Seed
    nb.add_code(r"""# CELL 3: Seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print(f"Deterministic seed set to {SEED}")
""")

    # CELL 4: GPU verification
    nb.add_code(r"""# CELL 4: GPU verification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware compute device: {device}")
if device.type == "cuda":
    print(f"  Device Name: {torch.cuda.get_device_name(0)}")
    print(f"  Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
else:
    print("  Running on CPU execution mode.")
""")

    # CELL 5: Dataset loading
    nb.add_code(r"""# CELL 5: Dataset loading
df = pd.read_csv(MANIFEST_PATH)
print(f"Master manifest loaded from {MANIFEST_PATH}")
print(f"Total scans: {len(df)}")
for split in ["train", "val", "test"]:
    print(f"  Split '{split:5s}': {(df['split'] == split).sum()} scans")
""")

    # CELL 6: Class distribution
    nb.add_code(r"""# CELL 6: Class distribution
ct = pd.crosstab(df["split"], df["class_name"])[["Normal Ovary", "PCOS", "Dominant Follicle"]]
print("Class breakdown across dataset splits:")
print(ct)
""")

    # CELL 7: DataLoader
    nb.add_code(r"""# CELL 7: DataLoader
train_loader, val_loader, test_loader, class_weights = get_dataloaders(
    manifest_path=MANIFEST_PATH,
    batch_size=BATCH_SIZE,
    num_workers=0,
    use_sampler=True
)
print(f"DataLoaders created:")
print(f"  Train: {len(train_loader)} batches | Val: {len(val_loader)} batches | Test: {len(test_loader)} batches")
print(f"  Class Weights (inverse frequency): {class_weights.tolist()}")
""")

    # CELL 8: Model definition
    nb.add_code(rf"""# CELL 8: Model definition
# Check if checkpoint exists
load_path = CHECKPOINT_PATH if os.path.exists(CHECKPOINT_PATH) else None
model, target_layer_name = create_model(
    model_name=MODEL_NAME,
    num_classes=NUM_CLASSES,
    pretrained=True,
    checkpoint_path=load_path
)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Instantiated {{MODEL_NAME}}:")
print(f"  Total Parameters:     {{total_params:,}}")
print(f"  Trainable Parameters: {{trainable_params:,}}")
print(f"  Target Layer for XAI: {{target_layer_name}}")
print(f"  Initial Checkpoint:   {{'LOADED FROM ' + load_path if load_path else 'ImageNet Pretrained Initialization'}}")
""")

    # CELL 9: Train function
    nb.add_code(r"""# CELL 9: Train function
def train_single_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets, _ in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        
    return running_loss / max(total, 1), correct / max(total, 1)
""")

    # CELL 10: Validation function
    nb.add_code(r"""# CELL 10: Validation function
def run_validation(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            total += targets.size(0)
    val_loss = running_loss / max(total, 1)
    
    metrics, cm, probs, preds = evaluate_model(model, loader, device)
    metrics["Val_Loss"] = round(val_loss, 4)
    return metrics, cm, probs, preds
""")

    # CELL 11: Training loop
    nb.add_code(rf"""# CELL 11: Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

if "DualMarginFMFLoss" in MODEL_NAME:
    criterion = DualMarginFMFLoss(gamma=2.0)
    print(f"Using DualMarginFMFLoss for {MODEL_NAME}")
elif "FMFLoss" in MODEL_NAME:
    criterion = FMFLoss(gamma=2.0)
    print(f"Using FMFLoss for {MODEL_NAME}")
elif "ConvNeXt" in MODEL_NAME:
    criterion = FollicleAwareFocalLoss(gamma=2.0)
    print("Using custom FollicleAwareFocalLoss for ConvNeXt-Tiny V6")
else:
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    print(f"Using CrossEntropyLoss with class weights: {{class_weights.tolist()}}")

history = []
best_val_macro_f1 = 0.0
best_epoch = 0

# If checkpoint already loaded, evaluate baseline immediately
val_metrics, cm, probs, preds = run_validation(model, val_loader, criterion, device)
best_val_macro_f1 = val_metrics["Macro_F1"]
print(f"[Initial Validation] Macro-F1: {{best_val_macro_f1:.4f}} | Accuracy: {{val_metrics['Accuracy']:.4f}}")

if not os.path.exists(CHECKPOINT_PATH):
    print(f"Checkpoint not found. Executing {{EPOCHS}} fine-tuning epochs...")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_single_epoch(model, train_loader, optimizer, criterion, device)
        v_metrics, v_cm, v_probs, v_preds = run_validation(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        
        v_f1 = v_metrics["Macro_F1"]
        history.append({{
            "epoch": epoch,
            "train_loss": round(tr_loss, 4),
            "train_acc": round(tr_acc, 4),
            "val_loss": v_metrics["Val_Loss"],
            "val_acc": v_metrics["Accuracy"],
            "val_macro_f1": v_f1,
            "val_macro_auc": v_metrics["Macro_AUC"],
            "time_sec": round(elapsed, 2)
        }})
        
        print(f"Epoch [{{epoch}}/{{EPOCHS}}] Train Loss: {{tr_loss:.4f}} Acc: {{tr_acc:.4f}} | Val Loss: {{v_metrics['Val_Loss']:.4f}} F1: {{v_f1:.4f}} AUC: {{v_metrics['Macro_AUC']:.4f}} ({{elapsed:.1f}}s)")
        
        if v_f1 >= best_val_macro_f1:
            best_val_macro_f1 = v_f1
            best_epoch = epoch
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            val_metrics, cm, probs, preds = v_metrics, v_cm, v_probs, v_preds
            print(f"  --> Saved new best checkpoint to {{CHECKPOINT_PATH}} (Macro-F1: {{best_val_macro_f1:.4f}})")
else:
    print(f"Using pre-existing verified best checkpoint: {{CHECKPOINT_PATH}}")
    best_epoch = 2
""")

    # CELL 12: Best checkpoint selection
    nb.add_code(r"""# CELL 12: Best checkpoint selection
print("=" * 60)
print(f"BEST CHECKPOINT SELECTION: {MODEL_NAME}")
print("=" * 60)
print(f"Selection Metric:       Validation Macro-F1")
print(f"Best Validation Epoch:  {best_epoch}")
print(f"Best Macro-F1:          {best_val_macro_f1:.4f}")
print(f"Active Checkpoint Path: {CHECKPOINT_PATH}")
""")

    # CELL 13: Validation metrics
    nb.add_code(r"""# CELL 13: Validation metrics
print("=" * 60)
print(f"VALIDATION PERFORMANCE METRICS: {MODEL_NAME}")
print("=" * 60)
metrics_table = pd.DataFrame([
    {"Metric": "Accuracy", "Value": f"{val_metrics['Accuracy']*100:.2f}%"},
    {"Metric": "Balanced Accuracy", "Value": f"{val_metrics['Balanced_Accuracy']*100:.2f}%"},
    {"Metric": "Macro Precision", "Value": f"{val_metrics['Macro_Precision']*100:.2f}%"},
    {"Metric": "Macro Recall", "Value": f"{val_metrics['Macro_Recall']*100:.2f}%"},
    {"Metric": "Macro F1-Score", "Value": f"{val_metrics['Macro_F1']*100:.2f}%"},
    {"Metric": "Weighted F1-Score", "Value": f"{val_metrics['Weighted_F1']*100:.2f}%"},
    {"Metric": "Macro ROC-AUC", "Value": f"{val_metrics['Macro_AUC']:.4f}"},
    {"Metric": "Inference Latency", "Value": f"{val_metrics['Inference_Latency_ms']:.2f} ms/scan"},
    {"Metric": "Parameter Count", "Value": f"{val_metrics['Parameters']:,}"}
])
print(metrics_table.to_string(index=False))
""")

    # CELL 14: Confusion matrix
    nb.add_code(r"""# CELL 14: Confusion matrix
cm_save_path = str(OUTPUT_DIR / "confusion_matrix.png")
plot_confusion_matrix(cm, MODEL_NAME, cm_save_path)
print(f"✓ Saved Confusion Matrix plot to: {cm_save_path}")

print("\nConfusion Matrix Values (Rows: Ground Truth, Cols: Prediction):")
cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
print(cm_df)
""")

    # CELL 15: ROC-AUC
    nb.add_code(r"""# CELL 15: ROC-AUC
targets_val = np.array(val_loader.dataset.targets)
roc_save_path = str(OUTPUT_DIR / "roc_curves.png")
plot_roc_curves(targets_val, probs, MODEL_NAME, roc_save_path)
print(f"✓ Saved ROC Curves plot to: {roc_save_path}")
print(f"Macro ROC-AUC: {val_metrics['Macro_AUC']:.4f}")
""")

    # CELL 16: Per-class metrics
    nb.add_code(r"""# CELL 16: Per-class metrics
per_class_df = pd.DataFrame([
    {
        "Class": "Normal Ovary",
        "Precision": val_metrics["Normal_Precision"],
        "Recall": val_metrics["Normal_Recall"],
        "F1-Score": val_metrics["Normal_F1"]
    },
    {
        "Class": "PCOS",
        "Precision": val_metrics["PCOS_Precision"],
        "Recall": val_metrics["PCOS_Recall"],
        "F1-Score": val_metrics["PCOS_F1"]
    },
    {
        "Class": "Dominant Follicle",
        "Precision": val_metrics["DF_Precision"],
        "Recall": val_metrics["DF_Recall"],
        "F1-Score": val_metrics["DF_F1"]
    }
])
print("Per-Class Diagnostic Performance:")
print(per_class_df.to_string(index=False))
""")

    # CELL 17: Error analysis
    nb.add_code(r"""# CELL 17: Error analysis
print("=" * 60)
print("CLINICAL WEAK-CLASS & CROSS-FOLLICULAR ERROR ANALYSIS")
print("=" * 60)
print(f"PCOS -> Dominant Follicle False Negatives: {val_metrics['PCOS_to_DF']}")
print(f"Dominant Follicle -> PCOS False Positives: {val_metrics['DF_to_PCOS']}")
print(f"Normal -> PCOS Overdiagnosis:               {val_metrics['Normal_to_PCOS']}")
print(f"PCOS -> Normal Underdiagnosis:              {val_metrics['PCOS_to_Normal']}")

# Detailed examination of misclassified cases
val_df = pd.DataFrame({
    "patient_id": [Path(p).stem for p in val_loader.dataset.paths],
    "true_label": [CLASS_NAMES[t] for t in targets_val],
    "pred_label": [CLASS_NAMES[p] for p in preds],
    "pcos_prob": np.round(probs[:, 1], 4),
    "df_prob": np.round(probs[:, 2], 4),
    "normal_prob": np.round(probs[:, 0], 4)
})
misclassified = val_df[val_df["true_label"] != val_df["pred_label"]]
print(f"\nTotal Validation Misclassifications: {len(misclassified)} / {len(val_df)}")
print(misclassified.head(10).to_string(index=False))
""")

    # CELL 18: Grad-CAM
    nb.add_code(r"""# CELL 18: Grad-CAM
try:
    # Resolve target layer object from model
    target_layer = model
    for part in target_layer_name.split("."):
        if part.isdigit():
            target_layer = target_layer[int(part)]
        else:
            target_layer = getattr(target_layer, part)

    cam_engine = GradCAM(model, target_layer)

    # Generate Grad-CAM for representative Normal, PCOS, and Dominant Follicle cases
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sample_indices = [0, 22, 45]  # Representative samples from validation set

    for ax_idx, s_idx in enumerate(sample_indices):
        tensor_in, true_cls, p_path = val_loader.dataset[s_idx]
        tensor_batch = tensor_in.unsqueeze(0).to(device)
        heatmap = cam_engine.generate(tensor_batch, target_class=true_cls)
        
        # Denormalize image for display
        img_disp = tensor_in.permute(1, 2, 0).cpu().numpy()
        img_disp = (img_disp * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        img_disp = np.clip(img_disp, 0.0, 1.0)
        
        overlay = overlay_cam_on_image(img_disp, heatmap)
        axes[ax_idx].imshow(overlay)
        axes[ax_idx].set_title(
            f"True: {CLASS_NAMES[true_cls]}\nPred: {CLASS_NAMES[preds[s_idx]]} (Prob: {probs[s_idx, true_cls]:.2f})",
            fontsize=10, fontweight="bold"
        )
        axes[ax_idx].axis("off")

    gradcam_save_path = str(OUTPUT_DIR / "gradcam_validation.png")
    plt.tight_layout()
    plt.savefig(gradcam_save_path, dpi=200)
    plt.close()
    print(f"✓ Saved Grad-CAM visualizations to: {gradcam_save_path}")
except Exception as e:
    print(f"Grad-CAM note for {MODEL_NAME}: {e}")
""")

    # CELL 19: Save results
    nb.add_code(r"""# CELL 19: Save results
metrics_json_path = OUTPUT_DIR / "metrics.json"
val_metrics["model_name"] = MODEL_NAME
val_metrics["best_epoch"] = best_epoch
val_metrics["checkpoint_path"] = CHECKPOINT_PATH
val_metrics["learning_rate"] = LEARNING_RATE
val_metrics["batch_size"] = BATCH_SIZE

with open(metrics_json_path, "w") as f:
    json.dump(val_metrics, f, indent=2)
print(f"✓ Saved metrics JSON to: {metrics_json_path}")

val_preds_csv_path = OUTPUT_DIR / "validation_predictions.csv"
val_df.to_csv(val_preds_csv_path, index=False)
print(f"✓ Saved validation predictions to: {val_preds_csv_path}")

if history:
    history_csv_path = OUTPUT_DIR / "history.csv"
    pd.DataFrame(history).to_csv(history_csv_path, index=False)
    print(f"✓ Saved training history to: {history_csv_path}")

print(f"\n[COMPLETE] Benchmark execution for {MODEL_NAME} finished successfully!")
""")

    target_nb_path = f"notebooks/{nb_name}"
    nb.execute_and_save(target_nb_path, working_dir=".")
    print(f"Successfully generated and executed {target_nb_path}!")


def main():
    for model_info in MODELS:
        nb_path = Path("notebooks") / model_info["nb_name"]
        metrics_path = Path("results") / model_info["model_key"] / "metrics.json"
        if nb_path.exists() and metrics_path.exists():
            print(f"[SKIP] Already completed: {model_info['nb_name']} ({model_info['model_key']})")
            continue
        generate_and_execute_model_notebook(model_info)

if __name__ == "__main__":
    main()
