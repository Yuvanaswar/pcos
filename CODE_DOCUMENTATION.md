# Code Documentation & Architecture Guide
## 3-Class Ovarian Ultrasound Classification Benchmark

**Project Root:** `d:/pcos project test/`  
**Target Diagnostics:** Class 0 = Normal Ovary, Class 1 = PCOS / PCO, Class 2 = Dominant Follicle  
**Runtime:** Python 3.10+ (PyTorch 2.0+, OpenCV, Torchvision)  

---

## 1. Directory Tree & Architecture Map

```
d:/pcos project test/
├── config.py                               # Global paths, class maps, hyperparams
├── manifest.csv                            # Master 571-image cryptographic manifest
├── FINAL_COMPARISON.csv                    # Single-pass locked test benchmark results
├── FINAL_RESEARCH_REPORT.md                # 15 clinical research questions & conclusions
├── PROJECT_REPORT.md                       # 10-point academic project report
├── CODE_DOCUMENTATION.md                   # This comprehensive technical code guide
│
├── data/                                   # Audited ultrasound dataset (571 scans)
│   ├── train/                              # 400 images (DF: 148, NORMAL: 155, PCOS: 97)
│   ├── val/                                # 85 images (DF: 32, NORMAL: 32, PCOS: 21)
│   └── test/                               # 86 images (DF: 31, NORMAL: 34, PCOS: 21)
│
├── src/                                    # Modular python library
│   ├── __init__.py                         # Package initialization
│   ├── preprocessing.py                    # Letterbox padding & Telea inpainting
│   ├── dataset.py                          # PyTorch Dataset, Loader, WeightedSampler
│   ├── losses.py                           # FollicleAwareFocalLoss implementation
│   ├── evaluate.py                         # Evaluation metrics, confusion matrix, ROC-AUC
│   ├── gradcam.py                          # Grad-CAM spatial attribution engine
│   ├── nb_utils.py                         # Programmatic Jupyter Notebook builder & runner
│   ├── models/
│   │   ├── __init__.py
│   │   └── factory.py                      # Unified model registry across 8 backbones
│   └── segmentation/
│       ├── __init__.py
│       ├── unet.py                         # U-Net & Attention U-Net architectures
│       └── roi_extractor.py                # Parenchymal bounding box & crop extractor
│
├── notebooks/                              # 15 Executed & verified Jupyter Notebooks
│   ├── 01_DATA_AUDIT.ipynb                 # Audit, SHA-256 checks, leakage verification
│   ├── 02_IMAGE_AUDIT.ipynb                # Calipers, borders, dynamic range screening
│   ├── 03_COMMON_PREPROCESSING.ipynb       # Aspect padding & Telea inpainting pipeline
│   ├── 04_RESNET50.ipynb                   # ResNet-50 benchmark
│   ├── 05_DENSENET121.ipynb                # DenseNet-121 benchmark
│   ├── 06_VGG16.ipynb                      # VGG-16 with BN benchmark
│   ├── 07_MOBILENETV2.ipynb                # MobileNet-V2 benchmark
│   ├── 08_EFFICIENTNETB0.ipynb             # EfficientNet-B0 benchmark
│   ├── 09_CONVNEXT_TINY_V6.ipynb           # ConvNeXt-Tiny V6 benchmark
│   ├── 10_SWIN_TINY.ipynb                  # Swin Transformer Tiny benchmark
│   ├── 11_VIT_B16.ipynb                    # Vision Transformer ViT-B/16 benchmark
│   ├── 12_MASTER_MODEL_COMPARISON.ipynb    # Multi-architecture validation comparison
│   ├── 13_ROI_SEGMENTATION_ABLATION.ipynb  # U-Net vs MobileSAM vs Direct ablation
│   ├── 14_XAI_COMPARISON.ipynb             # Grad-CAM visual heatmaps & leakage check
│   └── 15_ABLATION_STUDY.ipynb             # A0 to A6 component isolation study
│
├── build_01_data_audit.py                  # Generator script for notebook 01
├── build_02_image_audit.py                 # Generator script for notebook 02
├── build_03_preprocessing.py               # Generator script for notebook 03
├── build_all_model_notebooks.py            # Generator & training script for notebooks 04-11
├── build_12_master_comparison.py           # Generator script for notebook 12
├── build_13_roi_ablation.py                # Generator script for notebook 13
├── build_14_xai.py                         # Generator script for notebook 14
├── build_15_ablation.py                    # Generator script for notebook 15
├── run_final_locked_test.py                # Single-pass locked test evaluation runner
├── generate_all_segmented_images.py        # Generates all 571 crops and overlays
├── fix_all_notebooks.py                    # Jupyter cell ID & schema compliance utility
│
└── results/
    ├── checkpoints/                        # 8 trained PyTorch weights (.pth)
    ├── figures/                            # High-resolution benchmark figures
    ├── image_quality_audit.csv             # Artifact screening metrics per scan
    ├── master_model_comparison.csv         # Validation comparison across 8 backbones
    ├── pcos_specific_comparison.csv        # Detailed PCOS recall & cross-errors
    ├── roi_ablation_results.csv            # ROI localization downstream ablation
    ├── ablation_study_results.csv          # A0 to A6 component isolation deltas
    └── segmented_images/
        ├── crops/                          # 571 parenchymal ROI crops (224x224)
        └── overlays/                       # 571 diagnostic green contour overlays
```

---

## 2. Global Configuration Module (`config.py`)

[config.py](file:///d:/pcos%20project%20test/config.py) provides single-source-of-truth constants for all modules:

### Key Constants & Mappings
- `BASE_DIR`, `DATA_DIR`, `RESULTS_DIR`, `CHECKPOINTS_DIR`, `NOTEBOOKS_DIR`, `FIGURES_DIR`: System path objects.
- `CLASS_MAP`:
  ```python
  CLASS_MAP = {"Normal Ovary": 0, "PCOS": 1, "Dominant Follicle": 2}
  ```
- `FOLDER_NAME_MAP`: Maps folder aliases (e.g., `NORMAL`, `Normal_preservation`, `EPS`, `PCO`, `pcos`, `DF`, `dominant_follicle`) to canonical diagnostic names.
- `IMAGE_SIZE = (224, 224)`: Standardized benchmark input resolution.
- `BATCH_SIZE = 16`, `RANDOM_SEED = 42`.
- `IMAGENET_MEAN = [0.485, 0.456, 0.406]`, `IMAGENET_STD = [0.229, 0.224, 0.225]`.

---

## 3. Core Source Library (`src/`)

### 3.1 Preprocessing Engine (`src/preprocessing.py`)
[src/preprocessing.py](file:///d:/pcos%20project%20test/src/preprocessing.py) ensures aspect-ratio preservation and suppresses artificial measurement shortcuts.

#### `letterbox_image(image, target_size=(224, 224), pad_value=0)`
- **Purpose:** Resizes an image without stretching by scaling the longer dimension to `target_size` and applying symmetric zero-padding along the shorter dimension.
- **Returns:** `(padded_image, scale_factor, (pad_left, pad_top))`.

#### `inpaint_calipers(image, threshold=250, min_area=6, max_area=60)`
- **Purpose:** Detects white measurement crosshairs ($\text{intensity} \ge 250$) and executes the OpenCV Telea Fast Marching Method (`cv2.INPAINT_TELEA`, radius=3) to smoothly restore ovarian parenchyma.

#### `UltrasoundPreprocessingPipeline(target_size=(224, 224), is_train=False, inpaint=True)`
- **Callable class:** Integrates letterboxing, caliper inpainting, conservative training augmentations (horizontal flip $p=0.5$, rotation $\pm 7^\circ$, gain variation $\pm 10\%$), and ImageNet tensor normalization.

---

### 3.2 Dataset & DataLoaders (`src/dataset.py`)
[src/dataset.py](file:///d:/pcos%20project%20test/src/dataset.py) manages PyTorch dataset loading and class balancing.

#### `OvarianUltrasoundDataset(manifest_path, split="train", is_train=None, inpaint=True)`
- **Inherits:** `torch.utils.data.Dataset`.
- **Functionality:** Filters `manifest.csv` by split (`train`, `val`, `test`), reads the raw image from disk, applies `UltrasoundPreprocessingPipeline`, and returns `(tensor, label, image_path)`.

#### `get_dataloaders(manifest_path, batch_size=16, num_workers=0, use_weighted_sampler=True)`
- **Class Balancing:** Computes inverse class frequencies and builds a `WeightedRandomSampler` for the training partition:
  $$w_c = \frac{N_{\text{total}}}{C \cdot N_c}$$
- **Determinism:** Validation and test loaders use sequential deterministic loaders (`shuffle=False`).

---

### 3.3 Loss Functions (`src/losses.py`)
[src/losses.py](file:///d:/pcos%20project%20test/src/losses.py) implements clinical loss formulations designed to prevent minority collapse and cross-follicular omissions.

#### `FollicleAwareFocalLoss(alpha=None, gamma=2.0, cross_follicle_penalty=2.0)`
- **Inherits:** `torch.nn.Module`.
- **Formulation:**
  $$\mathcal{L} = -\alpha_t (1 - p_t)^\gamma \log(p_t) + \lambda_{\text{cross}} \cdot \mathbb{I}(y \in \{1, 2\} \land \hat{y} \in \{1, 2\} \land y \neq \hat{y})$$
- **Clinical Rationale:** Suppresses easy negative gradients ($\gamma=2.0$) while doubling the loss penalty when a polycystic ovary ($y=1$) is misdiagnosed as a dominant follicle ($\hat{y}=2$), directly eliminating false-negative diagnostic omissions.

---

### 3.4 Model Factory (`src/models/factory.py`)
[src/models/factory.py](file:///d:/pcos%20project%20test/src/models/factory.py) encapsulates the creation, head adaptation, checkpoint loading, and layer introspection for all 8 benchmark backbones.

#### `MODEL_REGISTRY`
Dictionary mapping model keys to constructor functions, default pretrained weights, target feature layers for Grad-CAM, and classification head attributes:
- `ResNet50`: `layer4`, `fc` (2048 in)
- `DenseNet121`: `features.denseblock4`, `classifier` (1024 in)
- `VGG16`: `features.28`, `classifier.6` (4096 in)
- `MobileNetV2`: `features.18`, `classifier.1` (1280 in)
- `EfficientNetB0`: `features.8`, `classifier.1` (1280 in)
- `ConvNeXt_Tiny_V6`: `features.7`, `classifier.2` (768 in)
- `Swin_Tiny`: `norm`, `head` (768 in)
- `ViT_B16`: `encoder.ln`, `heads.head` (768 in)

#### `create_model(model_name, num_classes=3, pretrained=True, checkpoint_path=None)`
- Replaces the terminal linear projection head with a custom `nn.Linear(in_features, num_classes)`.
- Optionally loads fine-tuned `.pth` checkpoint weights onto CPU or GPU.
- **Returns:** `(model, target_cam_layer_name)`.

---

### 3.5 Segmentation Architectures (`src/segmentation/unet.py`)
[src/segmentation/unet.py](file:///d:/pcos%20project%20test/src/segmentation/unet.py) provides semantic segmentation architectures for automated ovarian localization.

#### `UNet(n_channels=3, n_classes=1)`
- Classic symmetrical encoder-decoder with lateral skip connections (`DoubleConv`, `Down`, `Up`, `OutConv`).

#### `AttentionUNet(n_channels=3, n_classes=1)`
- Integrates additive attention gates (`AttentionBlock`) along skip connections.
- Filters feature responses from lower encoder stages using the gating signal from higher stages:
  $$\alpha = \sigma(\psi^T(\text{ReLU}(W_x x + W_g g + b)))$$

---

### 3.6 ROI Extraction Engine (`src/segmentation/roi_extractor.py`)
[src/segmentation/roi_extractor.py](file:///d:/pcos%20project%20test/src/segmentation/roi_extractor.py) bridges segmentation and downstream classification.

#### `OvarianROIExtractor(method="unet", checkpoint_path=None)`
- **Methods Supported:** `"unet"`, `"attention_unet"`, `"mobilesam"`, `"medsam"`, `"morphological"`.
- **`extract_roi_crop(image_rgb)`:**
  1. Generates the parenchymal mask.
  2. Identifies the largest parenchymal contour.
  3. Computes the bounding box with an added **5% safety margin padding** to prevent clipping subcapsular peripheral follicles along the *tunica albuginea*.
  4. Crops and applies aspect-preserving letterboxing to $224 \times 224$.

#### `ROIEnhancedDataset(base_dataset, method="unet")`
- A dataset wrapper that intercepts items from `OvarianUltrasoundDataset`, passes the image through `OvarianROIExtractor`, and returns the letterboxed ROI crop tensor.

---

### 3.7 Explainable AI & Attributions (`src/gradcam.py`)
[src/gradcam.py](file:///d:/pcos%20project%20test/src/gradcam.py) implements gradient-weighted class activation mapping.

#### `GradCAM(model, target_layer_name)`
- Registers forward and backward hooks on the designated feature layer.
- Computes channel importance weights $\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial Y^c}{\partial A_{ij}^k}$.
- Generates the activation map $L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$.

#### `overlay_cam_on_image(img_rgb, cam_mask, alpha=0.5, colormap=cv2.COLORMAP_JET)`
- Normalizes and renders the jet colormap heatmap overlaid on the anatomical grayscale scan.

---

### 3.8 Evaluation Metrics & Reporting (`src/evaluate.py`)
[src/evaluate.py](file:///d:/pcos%20project%20test/src/evaluate.py) computes all benchmark metrics.

#### `evaluate_model(model, dataloader, device="cpu")`
- Calculates multi-class metrics:
  - Accuracy, Balanced Accuracy, Macro-Precision, Macro-Recall, Macro-F1, Weighted-F1
  - Macro ROC-AUC (One-vs-Rest)
  - Class-specific metrics: `Normal_F1`, `PCOS_F1`, `PCOS_Precision`, `PCOS_Recall`, `DF_F1`, `DF_Recall`
  - Cross-follicular confusion counts: `PCOS_to_DF`, `DF_to_PCOS`, `Normal_to_PCOS`, `PCOS_to_Normal`
- **Returns:** `(metrics_dict, confusion_matrix, probabilities, predictions)`.

#### `plot_confusion_matrix(...)`, `plot_roc_curves(...)`
- Generates seaborn annotated confusion matrices and scikit-learn One-vs-Rest ROC curves.

---

### 3.9 Programmatic Notebook Utilities (`src/nb_utils.py`)
[src/nb_utils.py](file:///d:/pcos%20project%20test/src/nb_utils.py) enables programmatic creation, execution, and export of Jupyter v4 notebooks.

#### `NotebookBuilder(title="...")`
- Methods: `add_markdown()`, `add_code()`, `to_dict()`, `save()`.
- **`execute_and_save(filepath, working_dir)`:** Executes all code cells sequentially in an isolated namespace, captures live stdout/stderr, populates notebook outputs, injects unique string `id` fields (UUID v4) on every cell, and writes schema-compliant Jupyter JSON (`nbformat: 4.4`).

---

## 4. Benchmark Construction & Execution Scripts

| Script | Purpose | Output |
| :--- | :--- | :--- |
| [build_01_data_audit.py](file:///d:/pcos%20project%20test/build_01_data_audit.py) | Audits data structure, computes SHA-256 hashes, checks leakage | `notebooks/01_DATA_AUDIT.ipynb`, `manifest.csv` |
| [build_02_image_audit.py](file:///d:/pcos%20project%20test/build_02_image_audit.py) | Screens resolution, aspect ratio, calipers, dynamic range | `notebooks/02_IMAGE_AUDIT.ipynb`, `results/image_quality_audit.csv` |
| [build_03_preprocessing.py](file:///d:/pcos%20project%20test/build_03_preprocessing.py) | Verifies letterboxing, inpainting, and data loaders | `notebooks/03_COMMON_PREPROCESSING.ipynb` |
| [build_all_model_notebooks.py](file:///d:/pcos%20project%20test/build_all_model_notebooks.py) | Trains & evaluates all 8 models (19 cells per notebook) | `notebooks/04_*.ipynb` to `11_*.ipynb`, `.pth` checkpoints |
| [build_12_master_comparison.py](file:///d:/pcos%20project%20test/build_12_master_comparison.py) | Aggregates validation metrics & hardware latency | `notebooks/12_MASTER_MODEL_COMPARISON.ipynb`, `results/*.csv` |
| [build_13_roi_ablation.py](file:///d:/pcos%20project%20test/build_13_roi_ablation.py) | Compares Direct vs U-Net vs Attention U-Net vs SAM | `notebooks/13_ROI_SEGMENTATION_ABLATION.ipynb`, `results/roi_*.csv` |
| [build_14_xai.py](file:///d:/pcos%20project%20test/build_14_xai.py) | Generates Grad-CAM visual heatmaps across models | `notebooks/14_XAI_COMPARISON.ipynb`, `results/figures/xai_*.png` |
| [build_15_ablation.py](file:///d:/pcos%20project%20test/build_15_ablation.py) | Runs component isolation study (A0 to A6) | `notebooks/15_ABLATION_STUDY.ipynb`, `results/ablation_*.csv` |
| [run_final_locked_test.py](file:///d:/pcos%20project%20test/run_final_locked_test.py) | Single-pass evaluation on 86 locked test scans | [FINAL_COMPARISON.csv](file:///d:/pcos%20project%20test/FINAL_COMPARISON.csv) |
| [generate_all_segmented_images.py](file:///d:/pcos%20project%20test/generate_all_segmented_images.py) | Generates parenchymal crops & green overlays for all 571 scans | `results/segmented_images/crops/`, `results/segmented_images/overlays/` |
| [fix_all_notebooks.py](file:///d:/pcos%20project%20test/fix_all_notebooks.py) | Normalizes cell IDs and schema compliance across all notebooks | Validated, instantly-savable `.ipynb` files |

---

## 5. Summary of Checkpoints & Generated Results

### Trained PyTorch Checkpoints (`results/checkpoints/`)
- `DenseNet121_best.pth` (27.1 MB)
- `MobileNetV2_best.pth` (8.7 MB)
- `ConvNeXt_Tiny_V6_best.pth` (106.2 MB)
- `ResNet50_best.pth` (90.0 MB)
- `VGG16_best.pth` (512.2 MB)
- `EfficientNetB0_best.pth` (15.6 MB)
- `Swin_Tiny_best.pth` (105.3 MB)
- `ViT_B16_best.pth` (327.3 MB)

### Generated Figures (`results/figures/`)
- `data_distribution.png`: Cross-split class balance chart and pie proportions.
- `image_audit_histograms.png`: Intensity distribution and dynamic range analysis.
- `artifact_examples.png`: Screening gallery of calipers, text, and acoustic padding.
- `preprocessing_pipeline.png`: Step-by-step visual letterbox and inpainting sequence.
- `master_model_comparison.png`: Macro-F1, Balanced Accuracy, and PCOS Recall comparison across models.
- `roi_ablation_comparison.png`: Direct vs U-Net vs MobileSAM classification performance.
- `xai_multimodel_comparison.png`: Grad-CAM parenchymal follicular attributions.
- `ablation_study_deltas.png`: Cumulative $\Delta$ gains from A0 baseline to A6 full pipeline.
- `all_segmented_samples.png`: Multi-class clinical gallery of parenchymal contours.

---

## 6. API Usage Example: Running Inference on an Ultrasound Scan

```python
import cv2
import torch
from src.models.factory import create_model
from src.preprocessing import UltrasoundPreprocessingPipeline

# 1. Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, _ = create_model(
    "MobileNetV2", 
    num_classes=3, 
    checkpoint_path="results/checkpoints/MobileNetV2_best.pth"
)
model.to(device).eval()

# 2. Preprocess scan
pipeline = UltrasoundPreprocessingPipeline(target_size=(224, 224), is_train=False, inpaint=True)
input_tensor = pipeline("data/test/PCOS/PCO_003.png").unsqueeze(0).to(device)

# 3. Predict class
with torch.no_grad():
    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

classes = ["Normal Ovary", "PCOS", "Dominant Follicle"]
predicted_idx = probs.argmax()
print(f"Prediction: {classes[predicted_idx]} (Confidence: {probs[predicted_idx]*100:.2f}%)")
print(f"Class Probabilities: Normal={probs[0]:.3f}, PCOS={probs[1]:.3f}, DF={probs[2]:.3f}")
```
