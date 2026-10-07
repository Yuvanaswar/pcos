"""
Common Preprocessing and Augmentation Pipeline for 3-Class Ovarian Ultrasound Benchmark
Strict adherence to:
- Image quality validation
- RGB conversion
- Aspect-ratio preserving letterbox resize (224x224) - NO STRETCHING
- Optional Telea caliper inpainting to remove spurious measurement markers
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Training augmentation: Horizontal flip (p=0.5), ±7° rotation, mild intensity gain (±10%)
- Validation/Test: Zero random augmentation (pure deterministic letterbox + norm)
"""

import math
import random
from typing import Tuple, Optional, Union
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def inpaint_calipers(img_rgb: np.ndarray, threshold: int = 250, radius: int = 3) -> np.ndarray:
    """
    Applies Telea fast-marching inpainting to remove electronic calipers, measurement crosses,
    and high-contrast marker lines to prevent artificial feature shortcut learning.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray >= threshold).astype(np.uint8) * 255
    
    # Filter for small connected components (crosses/calipers)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    caliper_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 4 <= area <= 100:  # Caliper / text marker size
            caliper_mask[labels == i] = 255
            
    if np.any(caliper_mask):
        inpainted = cv2.inpaint(img_rgb, caliper_mask, radius, cv2.INPAINT_TELEA)
        return inpainted
    return img_rgb


def letterbox_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    fill_color: Tuple[int, int, int] = (0, 0, 0)
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resizes an image preserving aspect ratio with symmetric zero-padding (letterboxing).
    NEVER stretches or distorts anatomical ovarian structures.
    
    Returns:
      letterboxed_image: (target_size[1], target_size[0], 3)
      scale: ratio applied to original image
      pad: (pad_w, pad_h) offsets
    """
    orig_h, orig_w = image.shape[:2]
    target_w, target_h = target_size

    # Calculate scale factor
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    # Resize with bilinear interpolation
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Compute symmetric padding
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    pad_w_extra = target_w - (new_w + pad_w)
    pad_h_extra = target_h - (new_h + pad_h)

    # Apply letterbox border
    letterboxed = cv2.copyMakeBorder(
        resized,
        pad_h, pad_h_extra, pad_w, pad_w_extra,
        cv2.BORDER_CONSTANT,
        value=fill_color
    )
    return letterboxed, scale, (pad_w, pad_h)


class UltrasoundPreprocessingPipeline:
    """
    Standardized, reproducible preprocessing pipeline for all models.
    """
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        is_train: bool = False,
        inpaint: bool = True
    ):
        self.target_size = target_size
        self.is_train = is_train
        self.inpaint = inpaint
        
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, img_input: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        # 1. Load / Convert to RGB NumPy array
        if isinstance(img_input, str):
            bgr = cv2.imread(img_input)
            if bgr is None:
                raise FileNotFoundError(f"Failed to load image from {img_input}")
            img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(img_input, Image.Image):
            img_rgb = np.array(img_input.convert("RGB"))
        elif isinstance(img_input, np.ndarray):
            if img_input.ndim == 2:
                img_rgb = cv2.cvtColor(img_input, cv2.COLOR_GRAY2RGB)
            elif img_input.shape[2] == 4:
                img_rgb = cv2.cvtColor(img_input, cv2.COLOR_RGBA2RGB)
            else:
                img_rgb = img_input
        else:
            raise TypeError(f"Unsupported image type: {type(img_input)}")

        # 2. Optional Telea Caliper Inpainting
        if self.inpaint:
            img_rgb = inpaint_calipers(img_rgb)

        # 3. Training Augmentation (Conservative, clinical)
        if self.is_train:
            # Horizontal Flip (p=0.5)
            if random.random() < 0.5:
                img_rgb = np.ascontiguousarray(np.fliplr(img_rgb))

            # Mild Rotation (±7 degrees)
            angle = random.uniform(-7.0, 7.0)
            h, w = img_rgb.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img_rgb = cv2.warpAffine(img_rgb, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

            # Mild Gain / Intensity Jitter (±10%)
            gain = random.uniform(0.90, 1.10)
            img_rgb = np.clip(img_rgb.astype(np.float32) * gain, 0, 255).astype(np.uint8)

        # 4. Aspect-ratio preserving resize & letterbox padding to (224, 224)
        letterboxed, _, _ = letterbox_image(img_rgb, target_size=self.target_size)

        # 5. Convert to Tensor (scales to [0, 1]) and apply ImageNet Normalization
        tensor = torch.from_numpy(letterboxed).permute(2, 0, 1).float() / 255.0
        normalized_tensor = self.normalize(tensor)

        return normalized_tensor
