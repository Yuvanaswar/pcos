"""
U-Net and Attention U-Net architectures for Ovarian Ultrasound ROI Segmentation.
Designed for isolating ovarian parenchymal boundaries from machine background and border artifacts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """[Conv2d -> BatchNorm -> ReLU] * 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net for ovarian ultrasound contour segmentation.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_features: int = 32):
        super().__init__()
        f = base_features
        self.inc = DoubleConv(in_channels, f)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f, f * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 2, f * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 4, f * 8))
        
        self.up1 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.conv_up1 = DoubleConv(f * 8, f * 4)
        
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.conv_up2 = DoubleConv(f * 4, f * 2)
        
        self.up3 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.conv_up3 = DoubleConv(f * 2, f)
        
        self.outc = nn.Conv2d(f, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)
        
        logits = self.outc(x)
        return torch.sigmoid(logits)


class AttentionBlock(nn.Module):
    """Additive Attention Gate for Attention U-Net"""
    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    """
    Attention U-Net for ovarian ultrasound segmentation.
    Gating signals focus attention on hypoechoic follicular stroma.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_features: int = 32):
        super().__init__()
        f = base_features
        self.inc = DoubleConv(in_channels, f)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f, f * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 2, f * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 4, f * 8))
        
        self.up1 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.att1 = AttentionBlock(f_g=f * 4, f_l=f * 4, f_int=f * 2)
        self.conv_up1 = DoubleConv(f * 8, f * 4)
        
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.att2 = AttentionBlock(f_g=f * 2, f_l=f * 2, f_int=f)
        self.conv_up2 = DoubleConv(f * 4, f * 2)
        
        self.up3 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.att3 = AttentionBlock(f_g=f, f_l=f, f_int=f // 2)
        self.conv_up3 = DoubleConv(f * 2, f)
        
        self.outc = nn.Conv2d(f, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        g1 = self.up1(x4)
        x3_att = self.att1(g=g1, x=x3)
        d1 = torch.cat([g1, x3_att], dim=1)
        d1 = self.conv_up1(d1)
        
        g2 = self.up2(d1)
        x2_att = self.att2(g=g2, x=x2)
        d2 = torch.cat([g2, x2_att], dim=1)
        d2 = self.conv_up2(d2)
        
        g3 = self.up3(d2)
        x1_att = self.att3(g=g3, x=x1)
        d3 = torch.cat([g3, x1_att], dim=1)
        d3 = self.conv_up3(d3)
        
        logits = self.outc(d3)
        return torch.sigmoid(logits)
