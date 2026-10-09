import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAMBlock(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

def inject_se_into_resnet(model):
    from torchvision.models.resnet import Bottleneck
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            for i, block in enumerate(module):
                if isinstance(block, Bottleneck):
                    # inject SE right after conv3/bn3 before addition
                    se = SEBlock(block.bn3.num_features)
                    # Register the module so it gets moved to GPU properly
                    setattr(block, 'se_block', se)
                    # We monkey-patch the forward method of the block
                    def new_forward(x, b=block):
                        identity = b.downsample(x) if b.downsample is not None else x
                        out = b.conv1(x)
                        out = b.bn1(out)
                        out = b.relu(out)
                        out = b.conv2(out)
                        out = b.bn2(out)
                        out = b.relu(out)
                        out = b.conv3(out)
                        out = b.bn3(out)
                        out = getattr(b, 'se_block')(out)  # INJECT SE HERE
                        out += identity
                        out = b.relu(out)
                        return out
                    block.forward = new_forward
    return model

def inject_cbam_into_resnet(model):
    from torchvision.models.resnet import Bottleneck
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            for i, block in enumerate(module):
                if isinstance(block, Bottleneck):
                    cbam = CBAMBlock(block.bn3.num_features)
                    # Register the module so it gets moved to GPU properly
                    setattr(block, 'cbam_block', cbam)
                    def new_forward(x, b=block):
                        identity = b.downsample(x) if b.downsample is not None else x
                        out = b.conv1(x)
                        out = b.bn1(out)
                        out = b.relu(out)
                        out = b.conv2(out)
                        out = b.bn2(out)
                        out = b.relu(out)
                        out = b.conv3(out)
                        out = b.bn3(out)
                        out = getattr(b, 'cbam_block')(out)  # INJECT CBAM HERE
                        out += identity
                        out = b.relu(out)
                        return out
                    block.forward = new_forward
    return model

def get_se_resnet50(pretrained=True):
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    return inject_se_into_resnet(model)

def get_cbam_resnet50(pretrained=True):
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    return inject_cbam_into_resnet(model)
