"""
损失函数模块

实现多种损失函数用于笔画优化，确保渲染结果与目标图像高度相似：
- L_pixel: L1 像素损失 - 确保颜色对齐
- L_perceptual: LPIPS 感知损失 - 确保语义对齐
- L_ssim: SSIM 结构相似性损失 - 保持结构信息
- L_color_hist: 颜色分布损失 - 保持整体色彩感觉
- L_length: 笔画长度正则化 - 防止扭曲笔画
- L_smooth: 笔画平滑度正则化 - 保持笔画流畅
- L_overlap: 笔画重叠惩罚 - 减少冗余笔画
"""

from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from brushes.base import BrushStroke


class PixelLoss(nn.Module):
    """L1/L2 像素损失"""
    
    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        self.loss_type = loss_type
        if loss_type == "l1":
            self.criterion = nn.L1Loss()
        elif loss_type == "l2":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if rendered.dim() == 3:
            rendered = rendered.unsqueeze(0)
            target = target.unsqueeze(0)
        return self.criterion(rendered, target)


class PerceptualLoss(nn.Module):
    """
    LPIPS 感知损失
    
    使用预训练的 VGG 网络提取特征，计算特征空间的差异。
    比像素损失更符合人类视觉感知，能捕捉语义差异。
    """
    
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        try:
            import lpips
            self.model = lpips.LPIPS(net='vgg').to(device)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
            self._available = True
        except ImportError:
            print("Warning: lpips not installed, perceptual loss will use MSE fallback")
            self._available = False
    
    def forward(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if rendered.dim() == 3:
            rendered = rendered.unsqueeze(0)
            target = target.unsqueeze(0)
        
        # LPIPS 需要 [-1, 1] 范围
        r = rendered * 2 - 1
        t = target * 2 - 1
        
        if self._available:
            return self.model(r, t).mean()
        else:
            # 回退到 MSE
            return F.mse_loss(r, t)


class SSIMLoss(nn.Module):
    """
    SSIM 结构相似性损失
    
    衡量两张图像在亮度、对比度、结构三个维度的相似性。
    比像素损失更符合人类视觉感知。
    
    SSIM(x, y) = (2*mu_x*mu_y + C1)(2*sigma_xy + C2) / (mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2)
    """
    
    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
    
    def _create_window(self, channels: int, device: str) -> torch.Tensor:
        """创建高斯窗口"""
        sigma = 1.5
        coords = torch.arange(self.window_size, dtype=torch.float32, device=device) - self.window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        
        window = g.unsqueeze(1) * g.unsqueeze(0)
        window = window.expand(channels, 1, self.window_size, self.window_size).contiguous()
        return window
    
    def forward(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if rendered.dim() == 3:
            rendered = rendered.unsqueeze(0)
            target = target.unsqueeze(0)
        
        channels = rendered.shape[1]
        window = self._create_window(channels, rendered.device)
        
        pad = self.window_size // 2
        
        mu_x = F.conv2d(rendered, window, padding=pad, groups=channels)
        mu_y = F.conv2d(target, window, padding=pad, groups=channels)
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.conv2d(rendered ** 2, window, padding=pad, groups=channels) - mu_x_sq
        sigma_y_sq = F.conv2d(target ** 2, window, padding=pad, groups=channels) - mu_y_sq
        sigma_xy = F.conv2d(rendered * target, window, padding=pad, groups=channels) - mu_xy
        
        ssim_map = ((2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)) / \
                   ((mu_x_sq + mu_y_sq + self.C1) * (sigma_x_sq + sigma_y_sq + self.C2))
        
        if self.size_average:
            ssim_val = ssim_map.mean()
        else:
            ssim_val = ssim_map.mean(dim=[1, 2, 3])
        
        # 返回 1 - SSIM 作为损失（SSIM 越高越好，损失越低越好）
        return 1 - ssim_val


class ColorHistogramLoss(nn.Module):
    """
    颜色分布损失
    
    确保渲染图像的整体颜色分布与目标图像一致。
    使用软直方图（可微）计算颜色分布差异。
    """
    
    def __init__(self, num_bins: int = 64):
        super().__init__()
        self.num_bins = num_bins
    
    def forward(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if rendered.dim() == 3:
            rendered_flat = rendered.reshape(3, -1).T  # [N, 3]
            target_flat = target.reshape(3, -1).T
        else:
            rendered_flat = rendered.reshape(rendered.shape[0], 3, -1).transpose(1, 2).reshape(-1, 3)
            target_flat = target.reshape(target.shape[0], 3, -1).transpose(1, 2).reshape(-1, 3)
        
        loss = 0.0
        for c in range(3):
            r_hist = self._soft_histogram(rendered_flat[:, c])
            t_hist = self._soft_histogram(target_flat[:, c])
            loss += F.l1_loss(r_hist, t_hist)
        
        return loss
    
    def _soft_histogram(self, x: torch.Tensor) -> torch.Tensor:
        """可微的软直方图"""
        bins = torch.linspace(0, 1, self.num_bins + 1, device=x.device)
        sigma = 1.0 / self.num_bins
        
        hist = torch.zeros(self.num_bins, device=x.device)
        for i in range(self.num_bins):
            center = (bins[i] + bins[i + 1]) / 2
            hist[i] = torch.exp(-((x - center) ** 2) / (2 * sigma ** 2)).mean()
        
        # 归一化
        hist = hist / (hist.sum() + 1e-8)
        return hist


class StrokeLengthLoss(nn.Module):
    """笔画长度正则化 - 防止过长的扭曲笔画"""
    
    def __init__(self, weight: float = 0.01):
        super().__init__()
        self.weight = weight
    
    def forward(self, strokes: List[BrushStroke]) -> torch.Tensor:
        if not strokes:
            return torch.tensor(0.0)
        total_length = sum(s.get_length() for s in strokes)
        return self.weight * total_length / len(strokes)


class StrokeSmoothnessLoss(nn.Module):
    """
    笔画平滑度正则化
    
    惩罚控制点之间的剧烈变化，使笔画更流畅自然。
    模拟人类画家的手部运动惯性。
    """
    
    def __init__(self, weight: float = 0.005):
        super().__init__()
        self.weight = weight
    
    def forward(self, strokes: List[BrushStroke]) -> torch.Tensor:
        if not strokes:
            return torch.tensor(0.0)
        
        total_smoothness = torch.tensor(0.0, device=strokes[0].raw_control_points.device)
        for stroke in strokes:
            cp = stroke.control_points
            if cp.shape[0] < 3:
                continue
            
            # 一阶差分（速度）
            velocity = cp[1:] - cp[:-1]
            # 二阶差分（加速度）
            acceleration = velocity[1:] - velocity[:-1]
            
            # 惩罚加速度（使笔画更平滑）
            total_smoothness += (acceleration ** 2).sum()
        
        return self.weight * total_smoothness / len(strokes)


class CombinedLoss(nn.Module):
    """
    组合损失函数
    
    L_total = λ_pixel * L_pixel + λ_perceptual * L_perceptual + λ_ssim * L_ssim
            + λ_color * L_color_hist + L_length + L_smooth
    
    权重根据优化阶段动态调整：
    - 早期：更重视像素损失和颜色分布
    - 后期：更重视感知损失和结构相似性
    """
    
    def __init__(
        self,
        device: str = "cpu",
        lambda_pixel: float = 1.0,
        lambda_perceptual: float = 10.0,
        lambda_ssim: float = 2.0,
        lambda_color: float = 1.0,
        lambda_length: float = 0.01,
        lambda_smooth: float = 0.005,
    ):
        super().__init__()
        self.lambda_pixel = lambda_pixel
        self.lambda_perceptual = lambda_perceptual
        self.lambda_ssim = lambda_ssim
        self.lambda_color = lambda_color
        self.lambda_length = lambda_length
        self.lambda_smooth = lambda_smooth
        
        self.pixel_loss = PixelLoss(loss_type="l1")
        self.perceptual_loss = PerceptualLoss(device=device)
        self.ssim_loss = SSIMLoss()
        self.color_loss = ColorHistogramLoss()
        self.length_loss = StrokeLengthLoss(weight=lambda_length)
        self.smoothness_loss = StrokeSmoothnessLoss(weight=lambda_smooth)
    
    def forward(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        strokes: List[BrushStroke],
        stage: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算组合损失。
        
        Args:
            rendered: 渲染图像 [C, H, W]
            target: 目标图像 [C, H, W]
            strokes: 笔画列表
            stage: 当前优化阶段（用于动态调整权重）
            
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的值
        """
        l_pixel = self.pixel_loss(rendered, target)
        l_perceptual = self.perceptual_loss(rendered, target)
        l_ssim = self.ssim_loss(rendered, target)
        l_color = self.color_loss(rendered, target)
        l_length = self.length_loss(strokes)
        l_smooth = self.smoothness_loss(strokes)
        
        # 根据阶段动态调整权重
        # 早期更重视像素对齐和颜色分布，后期更重视感知和结构
        if stage == 0:
            # 底色阶段：像素损失和颜色分布最重要
            w_pixel, w_perceptual, w_ssim, w_color = 2.0, 5.0, 0.5, 2.0
        elif stage == 1:
            # 刻画阶段：平衡各项损失
            w_pixel, w_perceptual, w_ssim, w_color = 1.0, 10.0, 2.0, 1.0
        else:
            # 细节阶段：感知损失和结构最重要
            w_pixel, w_perceptual, w_ssim, w_color = 0.5, 15.0, 3.0, 0.5
        
        total = (
            w_pixel * l_pixel
            + w_perceptual * l_perceptual
            + w_ssim * l_ssim
            + w_color * l_color
            + l_length
            + l_smooth
        )
        
        loss_dict = {
            "pixel": l_pixel.item(),
            "perceptual": l_perceptual.item(),
            "ssim": l_ssim.item(),
            "color_hist": l_color.item(),
            "length": l_length.item(),
            "smooth": l_smooth.item(),
            "total": total.item(),
        }
        
        return total, loss_dict
