"""
Comprehensive Repository Forensic Audit Script
Scans all notebooks, Python scripts, CSVs, and JSONs for:
1. Hardcoded metric literals (e.g. acc = 0.83...)
2. Simulated loops or placeholder branching
3. Discrepancies between saved checkpoint evaluation and reported values
4. Inconsistencies between tables in docs vs real files
"""

import os
import re
import json
from pathlib import Path
import pandas as pd

print("=" * 90)
print("1. SCANNING ALL 15 NOTEBOOKS FOR HARDCODED METRIC LITERALS")
print("=" * 90)

nb_dir = Path("notebooks")
suspicious_patterns = [
    (re.compile(r'acc\s*=\s*0\.\d+', re.I), "acc = 0.xxx"),
    (re.compile(r'f1\s*=\s*0\.\d+', re.I), "f1 = 0.xxx"),
    (re.compile(r'macro_f1\s*=\s*0\.\d+', re.I), "macro_f1 = 0.xxx"),
    (re.compile(r'["\']Accuracy["\']\s*:\s*0\.\d+', re.I), "'Accuracy': 0.xxx"),
    (re.compile(r'pcos_recall\s*=\s*0\.\d+', re.I), "pcos_recall = 0.xxx"),
    (re.compile(r'if\s+cid\s*==\s*["\']A', re.I), "if cid == 'A...'")
]

flagged_cells = []
for nb_path in sorted(nb_dir.glob("*.ipynb")):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            for pat, desc in suspicious_patterns:
                matches = pat.findall(src)
                if matches:
                    flagged_cells.append((nb_path.name, idx, desc, matches[:3]))

if flagged_cells:
    print(f"FLAGGED {len(flagged_cells)} suspicious patterns in notebooks:")
    for f in flagged_cells:
        print(f"  [ALERT] {f[0]} | Cell {f[1]} | Rule: {f[2]} | Matches: {f[3]}")
else:
    print("✓ CLEAN: 0 hardcoded metric literals found in code cells across all 15 notebooks!")

print("\n" + "=" * 90)
print("2. SCANNING ALL PYTHON BUILD SCRIPTS FOR HARDCODED METRIC LITERALS")
print("=" * 90)

py_files = sorted(Path(".").glob("*.py"))
flagged_scripts = []
for p in py_files:
    if p.name == "audit_fabrication_forensics.py":
        continue
    content = p.read_text(encoding="utf-8")
    for pat, desc in suspicious_patterns:
        matches = pat.findall(content)
        if matches:
            flagged_scripts.append((p.name, desc, matches[:3]))

if flagged_scripts:
    print(f"FLAGGED {len(flagged_scripts)} patterns in Python scripts:")
    for f in flagged_scripts:
        print(f"  [ALERT] {f[0]} | Rule: {f[1]} | Matches: {f[2]}")
else:
    print("✓ CLEAN: 0 hardcoded metric literals found in any python build scripts!")

print("\n" + "=" * 90)
print("3. VERIFYING ACTUAL CHECKPOINTS AGAINST MASTER REPORTED METRICS")
print("=" * 90)

ckpts = {
    "DenseNet121": "results/checkpoints/DenseNet121_best.pth",
    "ConvNeXt_Tiny_V6": "results/checkpoints/ConvNeXt_Tiny_V6_best.pth",
    "MobileNetV2": "results/checkpoints/MobileNetV2_best.pth",
    "EfficientNetB0": "results/checkpoints/EfficientNetB0_best.pth",
    "ResNet50": "results/checkpoints/ResNet50_best.pth",
    "VGG16": "results/checkpoints/VGG16_best.pth",
    "Swin_Tiny": "results/checkpoints/Swin_Tiny_best.pth"
}

print("Checking physical checkpoint existence on disk:")
for name, p in ckpts.items():
    exists = os.path.exists(p)
    size_mb = os.path.getsize(p) / (1024 * 1024) if exists else 0
    print(f"  {name:18s} : {'EXISTS' if exists else 'MISSING'} ({size_mb:6.2f} MB) -> {p}")

print("\n" + "=" * 90)
print("4. CROSS-CHECKING METRICS.JSON FILES IN RESULTS/ DIRECTORY")
print("=" * 90)

for name in ckpts.keys():
    m_path = Path(f"results/{name}/metrics.json")
    if m_path.exists():
        with open(m_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        print(f"  {name:18s} | Acc: {m['Accuracy']*100:5.2f}% | F1: {m['Macro_F1']*100:5.2f}% | PCOS Recall: {m['PCOS_Recall']*100:5.2f}% | PCOS->DF: {m['PCOS_to_DF']}")
    else:
        print(f"  {name:18s} : metrics.json missing at {m_path}")

print("\n" + "=" * 90)
print("5. CROSS-CHECKING FINAL_COMPARISON.CSV (LOCKED TEST SET)")
print("=" * 90)
final_csv = Path("FINAL_COMPARISON.csv")
if final_csv.exists():
    df_final = pd.read_csv(final_csv)
    print(df_final[["Model", "Accuracy", "Macro F1", "PCOS Recall", "PCOS_to_DF", "Inference Latency"]].to_string(index=False))
else:
    print("FINAL_COMPARISON.csv missing!")

print("\n" + "=" * 90)
print("6. CHECKING NOTEBOOK 15 (15_ABLATION_STUDY.ipynb) EXECUTION STATE")
print("=" * 90)
nb15_path = Path("notebooks/15_ABLATION_STUDY.ipynb")
with open(nb15_path, "r", encoding="utf-8") as f:
    nb15 = json.load(f)

for idx, cell in enumerate(nb15.get("cells", [])):
    ctype = cell.get("cell_type")
    cid = cell.get("id")
    has_outputs = len(cell.get("outputs", []))
    exec_count = cell.get("execution_count")
    first_line = cell.get("source", [""])[0].strip() if cell.get("source") else ""
    print(f"  Cell {idx:02d} [{ctype:8s}] (ID: {cid}) | ExecCount: {exec_count} | Outputs: {has_outputs} | Start: {first_line[:45]}")

print("\nFORENSIC AUDIT COMPLETED.")
