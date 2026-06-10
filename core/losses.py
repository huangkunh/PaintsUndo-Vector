"""
损失函数模块 - 高性能版本

优化：
1. VGG 感知损失：缓存目标特征，只计算渲染图前向
2. 颜色直方图：向量化实现，消除 Python 循环
3. SSIM：简化窗口计算
4. 可选轻量模式：只用像素损失，跳过 VGG
"""

from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from brushes.base import BrushStroke


class PixelLoss(nn.Module):
    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        if loss_type == "l1":
            self.criterion = nn.L1Loss(reduction='mean')
        elif loss_type == "l2":
            self.criterion = nn.MSELoss(reduction='mean')
        else:
            self.criterion = nn.SmoothL1Loss(reduction='mean')
    
    def forward(self, rendered, target):
        if rendered.dim() == 3:
            rendered = rendered.unsqueeze(0)
            target = target.unsqueeze(0)
        return self.criterion(rendered, target)


class VGGPerceptualLoss(nn.Module):
    """VGG 感知损失 - 缓存目标特征"""
    
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self._target_features = None
        self._target_hash = None
        
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        except:
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True)
        
        features = list(vgg.features.children())
        self.slice1 = nn.Sequential(*features[:4]).to(device).eval()
        self.slice2 = nn.Sequential(*features[4:9]).to(device).eval()
        self.slice3 = nn.Sequential(*features[9:16]).to(device).eval()
        
        for p in self.parameters():
            p.requires_grad = False
        
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _extract_features(self, x):
        x = (x - self.mean) / self.std
        f1 = self.slice1(x)
        f2 = self.slice2(f1)
        f3 = self.slice3(f2)
        return [f1, f2, f3]
    
    def cache_target(self, target: torch.Tensor):
        """缓存目标图像的特征，避免重复计算"""
        if target.dim() == 3:
            target = target.unsqueeze(0)
        with torch.no_grad():
            self._target_features = self._extract_features(target)
    
    def forward(self, rendered, target):
        if rendered.dim() == 3:
            rendered = rendered.unsqueeze(0)
            target = target.unsqueeze(0)
        
        r_feats = self._extract_features(rendered)
        
        # 使用缓存的目标特征
        if self._target_features is not None:
            t_feats = self._target_features
        else:
            with torch.no_grad():
                t_feats = self._extract_features(target)
        
        loss = torch.tensor(0.0, device=rendered.device)
        for r, t in zip(r_feats, t_feats):
            loss = loss + F.l1_loss(r, t)
        return loss / 3.0


class SSIMLoss(nn.Module):
    """SSIM 损失 - 简化版"""
    
    def __init__(self, window_size: int = 7, device: str = "cpu"):
        super().__init__()
        self.ws = window_size
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * (window_size / 6.0) ** 2))
        g = torch.outer(g, g)
        g = g / g.sum()
        self.register_buffer('_window', g.unsqueeze(0).unsqueeze(0))
    
    def forward(self, rendered, target):
        if rendered.dim() == 3:
            rendered = rendered.unsqueeze(0)
            target = target.unsqueeze(0)
        C = rendered.shape[1]
        w = self._window.expand(C, 1, -1, -1)
        pad = self.ws // 2
        
        mu_r = F.conv2d(rendered, w, padding=pad, groups=C)
        mu_t = F.conv2d(target, w, padding=pad, groups=C)
        
        sigma_r = F.conv2d(rendered ** 2, w, padding=pad, groups=C) - mu_r ** 2
        sigma_t = F.conv2d(target ** 2, w, padding=pad, groups=C) - mu_t ** 2
        sigma_rt = F.conv2d(rendered * target, w, padding=pad, groups=C) - mu_r * mu_t
        
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * mu_r * mu_t + C1) * (2 * sigma_rt + C2)) / \
               ((mu_r ** 2 + mu_t ** 2 + C1) * (sigma_r + sigma_t + C2))
        return 1.0 - ssim.mean()


class ColorHistogramLoss(nn.Module):
    """颜色分布损失 - 向量化实现"""
    
    def __init__(self, num_bins: int = 32, device: str = "cpu"):
        super().__init__()
        self.num_bins = num_bins
        centers = torch.linspace(0.5 / num_bins, 1 - 0.5 / num_bins, num_bins)
        self.register_buffer('_centers', centers)
        self._sigma = 1.0 / num_bins
    
    def _soft_hist(self, x: torch.Tensor) -> torch.Tensor:
        """向量化软直方图"""
        flat = x.reshape(-1).unsqueeze(1)  # [N, 1]
        centers = self._centers.unsqueeze(0)  # [1, B]
        hist = torch.exp(-0.5 * ((flat - centers) / self._sigma) ** 2).sum(dim=0)
        return hist / (hist.sum() + 1e-8)
    
    def forward(self, rendered, target):
        if rendered.dim() == 4:
            rendered = rendered[0]
            target = target[0]
        loss = torch.tensor(0.0, device=rendered.device)
        for c in range(3):
            loss = loss + F.l1_loss(self._soft_hist(rendered[c]), self._soft_hist(target[c]))
        return loss / 3.0


class StrokeLengthLoss(nn.Module):
    def __init__(self, weight: float = 0.01):
        super().__init__()
        self.weight = weight
    
    def forward(self, strokes):
        if not strokes:
            return torch.tensor(0.0, device=self.weight.device if hasattr(self.weight, 'device') else 'cpu')
        dev = strokes[0].control_points.device
        total = torch.tensor(0.0, device=dev)
        for s in strokes:
            d = s.control_points[1:] - s.control_points[:-1]
            total = total + d.pow(2).sum(dim=1).sqrt().sum()
        return self.weight * total / len(strokes)


class StrokeSmoothnessLoss(nn.Module):
    def __init__(self, weight: float = 0.005):
        super().__init__()
        self.weight = weight
    
    def forward(self, strokes):
        if not strokes:
            return torch.tensor(0.0)
        dev = strokes[0].control_points.device
        total = torch.tensor(0.0, device=dev)
        for s in strokes:
            cp = s.control_points
            if cp.shape[0] >= 3:
                d2 = (cp[2:] - 2 * cp[1:-1] + cp[:-2]).pow(2).sum(dim=1).mean()
                total = total + d2
        return self.weight * total / max(len(strokes), 1)


class CombinedLoss(nn.Module):
    """组合损失 - 支持轻量模式"""
    
    def __init__(self, device: str = "cpu", lightweight: bool = False,
                 lambda_pixel=1.0, lambda_perceptual=10.0, lambda_ssim=2.0,
                 lambda_color=1.0, lambda_length=0.01, lambda_smooth=0.005):
        super().__init__()
        self.lightweight = lightweight
        self.device = device
        
        self.pixel_loss = PixelLoss("l1")
        self.length_loss = StrokeLengthLoss(lambda_length)
        self.smoothness_loss = StrokeSmoothnessLoss(lambda_smooth)
        
        if not lightweight:
            self.perceptual_loss = VGGPerceptualLoss(device=device)
            self.ssim_loss = SSIMLoss(device=device)
            self.color_loss = ColorHistogramLoss(device=device)
        else:
            self.perceptual_loss = None
            self.ssim_loss = None
            self.color_loss = None
    
    def cache_target(self, target: torch.Tensor):
        """缓存目标特征（VGG）"""
        if self.perceptual_loss is not None:
            self.perceptual_loss.cache_target(target)
    
    def forward(self, rendered, target, strokes, stage=0):
        rendered = rendered.clamp(0.0, 1.0)
        
        if torch.isnan(rendered).any():
            zero = rendered.sum() * 0
            return zero, {"pixel": 1e3, "perceptual": 0, "ssim": 0, "color_hist": 0, "length": 0, "smooth": 0, "total": 1e3}
        
        l_pixel = self.pixel_loss(rendered, target)
        l_length = self.length_loss(strokes)
        l_smooth = self.smoothness_loss(strokes)
        
        if self.lightweight:
            # 轻量模式：只用像素损失 + 正则化
            total = l_pixel * 5.0 + l_length + l_smooth
            return total, {"pixel": l_pixel.item(), "perceptual": 0, "ssim": 0,
                           "color_hist": 0, "length": l_length.item(), "smooth": l_smooth.item(), "total": total.item()}
        
        l_perceptual = self.perceptual_loss(rendered, target)
        l_ssim = self.ssim_loss(rendered, target)
        l_color = self.color_loss(rendered, target)
        
        if stage == 0:
            wp, wpe, ws, wc = 2.0, 5.0, 0.5, 2.0
        elif stage == 1:
            wp, wpe, ws, wc = 1.0, 10.0, 2.0, 1.0
        else:
            wp, wpe, ws, wc = 0.5, 15.0, 3.0, 0.5
        
        total = wp * l_pixel + wpe * l_perceptual + ws * l_ssim + wc * l_color + l_length + l_smooth
        
        return total, {
            "pixel": l_pixel.item(), "perceptual": l_perceptual.item(),
            "ssim": l_ssim.item(), "color_hist": l_color.item(),
            "length": l_length.item(), "smooth": l_smooth.item(), "total": total.item()
        }
