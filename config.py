"""
Global Configuration for 3-Class Ovarian Ultrasound Benchmark
Classes:
  Class 0: Normal Ovary (Normal / Normal_preservation / EPS)
  Class 1: PCOS / PCO
  Class 2: Dominant Follicle (Dominant_Follicle / DF)
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
# Check if running in Google Colab with Drive mounted
if os.path.exists("/content/drive/MyDrive/pcos_data"):
    DATA_DIR = Path("/content/drive/MyDrive/pcos_data")
else:
    DATA_DIR = BASE_DIR / "data"

RESULTS_DIR = BASE_DIR / "results"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
FIGURES_DIR = RESULTS_DIR / "figures"

for d in [RESULTS_DIR, CHECKPOINTS_DIR, NOTEBOOKS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Diagnostic Target Classes
CLASS_MAP = {
    "Normal Ovary": 0,
    "PCOS": 1,
    "Dominant Follicle": 2
}
INDEX_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}
CLASS_NAMES = ["Normal Ovary", "PCOS", "Dominant Follicle"]
NUM_CLASSES = 3

# Folder names in raw/clean data
FOLDER_NAME_MAP = {
    # Class 0: Normal Ovary
    "NORMAL": "Normal Ovary",
    "Normal_preservation": "Normal Ovary",
    "EPS": "Normal Ovary",
    "Normal": "Normal Ovary",
    "normal": "Normal Ovary",
    # Class 1: PCOS
    "PCO": "PCOS",
    "PCOS": "PCOS",
    "pco": "PCOS",
    "pcos": "PCOS",
    # Class 2: Dominant Follicle
    "Dominant_Follicle": "Dominant Follicle",
    "dominant_follicle": "Dominant Follicle",
    "DF": "Dominant Follicle",
    "df": "Dominant Follicle"
}

# Image Preprocessing & Model Hyperparameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_WORKERS = 0
RANDOM_SEED = 42

# Normalization constants (ImageNet default)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
