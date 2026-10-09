"""
Model Factory for 3-Class Ovarian Ultrasound Classification Benchmark.
Supports:
  1. ResNet50
  2. DenseNet121
  3. VGG16
  4. MobileNetV2
  5. EfficientNetB0
  6. ConvNeXt_Tiny_V6
  7. Swin_Tiny
  8. ViT_B16
"""

import os
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchvision import models

MODEL_REGISTRY = {
    "ResNet50": {
        "constructor": models.resnet50,
        "weights": models.ResNet50_Weights.DEFAULT,
        "target_layer_name": "layer4",
        "head_attr": "fc",
        "in_features": 2048
    },
    "DenseNet121": {
        "constructor": models.densenet121,
        "weights": models.DenseNet121_Weights.DEFAULT,
        "target_layer_name": "features.denseblock4",
        "head_attr": "classifier",
        "in_features": 1024
    },
    "VGG16": {
        "constructor": models.vgg16,
        "weights": models.VGG16_Weights.DEFAULT,
        "target_layer_name": "features.28",
        "head_attr": "classifier.6",
        "in_features": 4096
    },
    "MobileNetV2": {
        "constructor": models.mobilenet_v2,
        "weights": models.MobileNet_V2_Weights.DEFAULT,
        "target_layer_name": "features.18",
        "head_attr": "classifier.1",
        "in_features": 1280
    },
    "EfficientNetB0": {
        "constructor": models.efficientnet_b0,
        "weights": models.EfficientNet_B0_Weights.DEFAULT,
        "target_layer_name": "features.8",
        "head_attr": "classifier.1",
        "in_features": 1280
    },
    "ConvNeXt_Tiny_V6": {
        "constructor": models.convnext_tiny,
        "weights": models.ConvNeXt_Tiny_Weights.DEFAULT,
        "target_layer_name": "features.7",
        "head_attr": "classifier.2",
        "in_features": 768
    },
    "Swin_Tiny": {
        "constructor": models.swin_t,
        "weights": models.Swin_T_Weights.DEFAULT,
        "target_layer_name": "features.7",
        "head_attr": "head",
        "in_features": 768
    },
    "ViT_B16": {
        "constructor": models.vit_b_16,
        "weights": models.ViT_B_16_Weights.DEFAULT,
        "target_layer_name": "encoder.layers.encoder_layer_11.ln_1",
        "head_attr": "heads.head",
        "in_features": 768
    },
    "InceptionV3": {
        "constructor": lambda weights=None: models.inception_v3(weights=weights, aux_logits=False),
        "weights": models.Inception_V3_Weights.DEFAULT,
        "target_layer_name": "Mixed_7c",
        "head_attr": "fc",
        "in_features": 2048
    },
    "EfficientNetV2B0": {
        "constructor": models.efficientnet_v2_s,
        "weights": models.EfficientNet_V2_S_Weights.DEFAULT,
        "target_layer_name": "features.7",
        "head_attr": "classifier.1",
        "in_features": 1280
    },
    "SE-ResNet50": {
        "constructor": lambda weights=None: __import__('src.models.custom_resnet', fromlist=['']).get_se_resnet50(pretrained=(weights is not None)),
        "weights": models.ResNet50_Weights.DEFAULT,
        "target_layer_name": "layer4",
        "head_attr": "fc",
        "in_features": 2048
    },
    "CBAM-ResNet50": {
        "constructor": lambda weights=None: __import__('src.models.custom_resnet', fromlist=['']).get_cbam_resnet50(pretrained=(weights is not None)),
        "weights": models.ResNet50_Weights.DEFAULT,
        "target_layer_name": "layer4",
        "head_attr": "fc",
        "in_features": 2048
    },
    "SE-ResNet50-FMFLoss": {
        "constructor": lambda weights=None: __import__('src.models.custom_resnet', fromlist=['']).get_se_resnet50(pretrained=(weights is not None)),
        "weights": models.ResNet50_Weights.DEFAULT,
        "target_layer_name": "layer4",
        "head_attr": "fc",
        "in_features": 2048
    },
    "SE-ResNet50-DualMarginFMFLoss": {
        "constructor": lambda weights=None: __import__('src.models.custom_resnet', fromlist=['']).get_se_resnet50(pretrained=(weights is not None)),
        "weights": models.ResNet50_Weights.DEFAULT,
        "target_layer_name": "layer4",
        "head_attr": "fc",
        "in_features": 2048
    }
}


def create_model(
    model_name: str,
    num_classes: int = 3,
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None
) -> Tuple[nn.Module, str]:
    """
    Instantiates an architecture, configures its 3-class classification head,
    and loads weights/checkpoints if available.
    """
    cfg = MODEL_REGISTRY[model_name]
    weights = cfg["weights"] if pretrained else None
    
    # Check if weights file is locally cached to prevent multi-hour network download hangs
    hub_dir = torch.hub.get_dir()
    ckpt_dir = os.path.join(hub_dir, "checkpoints")
    is_cached = False
    if weights is not None:
        url = weights.url
        fname = os.path.basename(url)
        if os.path.exists(os.path.join(ckpt_dir, fname)):
            is_cached = True
    
    # If not cached locally, use weights=None for instant instantiation
    if not is_cached and model_name in ["Swin_Tiny", "ViT_B16"]:
        print(f"[{model_name}] Cached weights not found locally. Initializing architecture directly without blocking download.")
        weights = None

    try:
        model = cfg["constructor"](weights=weights)
    except Exception as e:
        print(f"[{model_name}] Online weights download unavailable ({e}). Initializing architecture.")
        model = cfg["constructor"](weights=None)

    # Adapt classification head
    in_feat = cfg["in_features"]
    head_attr = cfg["head_attr"]

    if "." in head_attr:
        parent_name, child_idx = head_attr.split(".")
        parent = getattr(model, parent_name)
        if child_idx.isdigit():
            parent[int(child_idx)] = nn.Linear(in_feat, num_classes)
        else:
            setattr(parent, child_idx, nn.Linear(in_feat, num_classes))
    else:
        setattr(model, head_attr, nn.Linear(in_feat, num_classes))

    # Load custom checkpoint if available
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[{model_name}] Loading checkpoint from: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model" in state:
                state = state["model"]
                
        cleaned = {}
        for k, v in state.items():
            clean_k = k.replace("module.", "").replace("backbone.", "")
            cleaned[clean_k] = v
            
        try:
            model.load_state_dict(cleaned, strict=False)
            print(f"[{model_name}] Checkpoint successfully loaded.")
        except Exception as e:
            print(f"[{model_name}] Warning loading checkpoint: {e}")

    return model, cfg["target_layer_name"]
