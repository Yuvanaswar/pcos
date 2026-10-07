"""
Explainable AI (Grad-CAM & LayerCAM) module for 3-Class Ovarian Ultrasound Benchmark
Supports CNNs (ResNet, DenseNet, VGG, MobileNet, EfficientNet, ConvNeXt) and Vision Transformers.
"""

from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        input_tensor: (1, 3, 224, 224)
        Returns: (224, 224) normalized heatmap in [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[0, target_class]
        score.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        # Handle 4D feature maps (B, C, H, W)
        if activations.dim() == 4:
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * activations, dim=1).squeeze(0)
        # Handle 3D token representations (B, Tokens, C) for ViT
        elif activations.dim() == 3:
            # Drop CLS token if present
            tokens = activations[:, 1:, :] if activations.shape[1] > 196 else activations
            grads = gradients[:, 1:, :] if gradients.shape[1] > 196 else gradients
            weights = torch.mean(grads, dim=1, keepdim=True)
            cam = torch.sum(weights * tokens, dim=2).squeeze(0)
            side = int(np.sqrt(cam.shape[0]))
            cam = cam.view(side, side)
        else:
            cam = activations.squeeze()

        cam = F.relu(cam)
        cam_np = cam.detach().cpu().numpy()
        
        # Normalize
        cam_max = np.max(cam_np)
        if cam_max > 0:
            cam_np = cam_np / cam_max
        else:
            cam_np = np.zeros_like(cam_np)

        cam_resized = cv2.resize(cam_np, (input_tensor.shape[3], input_tensor.shape[2]))
        return cam_resized


def overlay_cam_on_image(
    img_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.55,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlays a normalized Grad-CAM heatmap [0, 1] onto an RGB ultrasound image.
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored_cam = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_cam = cv2.cvtColor(colored_cam, cv2.COLOR_BGR2RGB)
    
    if img_rgb.max() <= 1.0:
        img_rgb = np.uint8(255 * img_rgb)
    elif img_rgb.dtype != np.uint8:
        img_rgb = img_rgb.astype(np.uint8)
        
    if img_rgb.shape[:2] != heatmap.shape[:2]:
        colored_cam = cv2.resize(colored_cam, (img_rgb.shape[1], img_rgb.shape[0]))

    overlay = cv2.addWeighted(img_rgb, 1 - alpha, colored_cam, alpha, 0)
    return overlay
