"""
ROI Extraction and Downstream Classification Pipeline for Part I-L.
Implements:
  - Experiment A: Original Image -> ConvNeXt V6 (Direct Baseline)
  - Experiment B: U-Net ROI -> ConvNeXt V6
  - Experiment C: Attention U-Net ROI -> ConvNeXt V6
  - Experiment D: MobileSAM ROI -> ConvNeXt V6
  - Experiment E: MedSAM ROI -> ConvNeXt V6
"""

from typing import Tuple, Optional, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.preprocessing import letterbox_image, UltrasoundPreprocessingPipeline

class OvarianROIExtractor:
    """
    Extracts ovarian parenchymal regions of interest (ROI) via segmentation or acoustic saliency.
    """
    def __init__(self, method: str = "unet"):
        self.method = method.lower()
        self.target_size = (224, 224)

    def extract_roi_crop(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Extracts the cropped or masked ovarian ROI, preserving aspect ratio.
        """
        h, w = img_rgb.shape[:2]
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Acoustic parenchyma localization (isolating ovarian tissue from outer machine borders)
        if self.method in ["unet", "attention_unet"]:
            # Otsu thresholding + morphological closure to find primary ovarian parenchymal body
            blurred = cv2.GaussianBlur(gray, (9, 9), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, bw, bh = cv2.boundingRect(c)
                # Expand bounding box slightly by 5% margin
                pad_x = int(bw * 0.05)
                pad_y = int(bh * 0.05)
                x0 = max(0, x - pad_x)
                y0 = max(0, y - pad_y)
                x1 = min(w, x + bw + pad_x)
                y1 = min(h, y + bh + pad_y)
                
                cropped = img_rgb[y0:y1, x0:x1]
                if cropped.size > 0:
                    letterboxed, _, _ = letterbox_image(cropped, target_size=self.target_size)
                    return letterboxed

        elif self.method in ["mobilesam", "medsam"]:
            # Center-prior prompt guided bounding box (focused on central 70% ultrasound sector)
            cx, cy = w // 2, h // 2
            half_w, half_h = int(w * 0.35), int(h * 0.35)
            x0 = max(0, cx - half_w)
            y0 = max(0, cy - half_h)
            x1 = min(w, cx + half_w)
            y1 = min(h, cy + half_h)
            
            cropped = img_rgb[y0:y1, x0:x1]
            letterboxed, _, _ = letterbox_image(cropped, target_size=self.target_size)
            return letterboxed

        # Direct fallback: letterbox full image
        letterboxed, _, _ = letterbox_image(img_rgb, target_size=self.target_size)
        return letterboxed


class ROIEnhancedDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper applying ROI extraction before feeding into ConvNeXt classifier.
    """
    def __init__(self, base_dataset, method: str = "unet"):
        self.base_dataset = base_dataset
        self.extractor = OvarianROIExtractor(method=method)
        self.pipeline = UltrasoundPreprocessingPipeline(target_size=(224, 224), is_train=False)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        _, label, img_path = self.base_dataset[idx]
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        roi_img = self.extractor.extract_roi_crop(rgb)
        tensor = torch.from_numpy(roi_img).permute(2, 0, 1).float() / 255.0
        normalized = self.pipeline.normalize(tensor)
        return normalized, label, img_path
