import os
import torch
import pandas as pd
from src.dataset import OvarianUltrasoundDataset
from torch.utils.data import DataLoader
from src.models.factory import create_model
from src.evaluate import evaluate_model, CLASS_NAMES

device = torch.device('cpu')
manifest = 'manifest.csv' if os.path.exists('manifest.csv') else 'results/manifest.csv'
val_ds = OvarianUltrasoundDataset(manifest, split='val', is_train=False)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

test_ds = OvarianUltrasoundDataset(manifest, split='test', is_train=False)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

ckpts = {
    'DenseNet121': 'results/checkpoints/DenseNet121_best.pth',
    'ConvNeXt_Tiny_V6': 'results/checkpoints/ConvNeXt_Tiny_V6_best.pth',
    'MobileNetV2': 'results/checkpoints/MobileNetV2_best.pth',
    'EfficientNetB0': 'results/checkpoints/EfficientNetB0_best.pth',
    'ResNet50': 'results/checkpoints/ResNet50_best.pth',
    'VGG16': 'results/checkpoints/VGG16_best.pth',
    'Swin_Tiny': 'results/checkpoints/Swin_Tiny_best.pth'
}

print("=" * 85)
print("ACTUAL LIVE VALIDATION EVALUATION OF REAL SAVED CHECKPOINTS (85 SCANS)")
print("=" * 85)
for name, ckpt in ckpts.items():
    if os.path.exists(ckpt):
        model, _ = create_model(name, num_classes=3, pretrained=False, checkpoint_path=ckpt)
        model.to(device).eval()
        m, cm, _, _ = evaluate_model(model, val_loader, device=device)
        print(f"{name:18s} | Acc: {m['Accuracy']*100:6.2f}% | F1: {m['Macro_F1']*100:6.2f}% | PCOS Rec: {m['PCOS_Recall']*100:6.2f}% | PCOS->DF: {m['PCOS_to_DF']} | CM: {cm.tolist()}")

print("\n" + "=" * 85)
print("ACTUAL LIVE TEST SET EVALUATION OF REAL SAVED CHECKPOINTS (86 UNSEEN SCANS)")
print("=" * 85)
for name, ckpt in ckpts.items():
    if os.path.exists(ckpt):
        model, _ = create_model(name, num_classes=3, pretrained=False, checkpoint_path=ckpt)
        model.to(device).eval()
        m, cm, _, _ = evaluate_model(model, test_loader, device=device)
        print(f"{name:18s} | Acc: {m['Accuracy']*100:6.2f}% | F1: {m['Macro_F1']*100:6.2f}% | PCOS Rec: {m['PCOS_Recall']*100:6.2f}% | PCOS->DF: {m['PCOS_to_DF']} | CM: {cm.tolist()}")
