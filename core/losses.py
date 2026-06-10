"""
损失函数模块

实现多种损失函数用于笔画优化，确保渲染结果与目标图像高度相似：
- L_pixel: L1/L2 像素损失 - 确保颜色对齐
- L_perceptual: VGG 感知损失（纯 PyTorch 实现，无需 lpips 库）
- L_ssim: SSIM 结构相似性损失 - 保持结构信息
- L_color_hist: 颜色分布损失 - 保持整体色彩感觉
- L_length: 笔画长度正则化 - 防止扭曲笔画
- L_smooth: 笔画平滑度正则化 - 保持笔画流畅
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
        if loss_type == "l1":
            self.criterion = nn.L1Loss(reduction='mean')
        elif loss_type == "l2":
            self.criterion = nn.MSELoss(reduction='mean')
        elif loss_type == "smooth_l1":
            self.criterion = nn.SmoothL1Loss(reduction='mean')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if rendered.dim() == 3:
            rendered = rendered.unsqueeze(0)
            target = target.unsqueeze(0)
        return self.criterion(rendered, target)


class VGGPerceptualLoss(nn.Module):
    """
    VGG 感知损失（纯 PyTorch 实现）
    
    使用 torchvision 预训练的 VGG16 提取多层特征，
    计算特征空间的 L1 差异。无需安装 lpips 库。
    
    特征层选择（模拟 LPIPS 的多层特征提取）：
    - relu1_2: 低级特征（边缘、纹理）
    - relu2_2: 中级特征（形状、模式）
    - relu3_3: 高级特征（语义、对象）
    """
    
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.device = device
        self.model = self._build_vgg().to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        
        # ImageNet 归一化参数
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))
    
    def _build_vgg(self) -> nn.Module:
        """构建 VGG16 特征提取器"""
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        except (ImportError, TypeError):
            try:
                from torchvision.models import vgg16, VGG16_Weights
                vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            except (ImportError, TypeError):
                from torchvision.models import vgg16
                vgg = vgg16(pretrained=True)
        
        # 提取 relu1_2, relu2_2, relu3_3 的特征
        features = vgg.features
        
        self.slice1 = nn.Sequential(*list(features.children())[:4])   # relu1_2.to(device)
        self.slice2 = nn.Sequential(*list(features.children())[4:9])  # relu2_2.to(device)
        self.slice3 = nn.Sequential(*list(features.children())[9:16]).to(self.device)
        
        return nn.Module()  # 占位，实际使用 slice
    
    def forward(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if rendered.dim() == 3:
            rendered = rendered.unsqueeze(0)
            target = target.unsqueeze(0)
        
        # 归一化
        r = (rendered - self.mean) / self.std
        t = (target - self.mean) / self.std
        
        # 多层特征提取
        loss = torch.tensor(0.0, device=rendered.device)
        
        # Layer 1
        r1 = self.slice1(r)
        t1 = self.slice1(t)
        loss = loss + F.l1_loss(r1, t1)
        
        # Layer 2
        r2 = self.slice2(r1)
        t2 = self.slice2(t1)
        loss = loss + F.l1_loss(r2, t2)
        
        # Layer 3
        r3 = self.slice3(r2)
        t3 = self.slice3(t2)
        loss = loss + F.l1_loss(r3, t3)
        
        return loss / 3.0


class LPIPSLoss(nn.Module):
    """
    LPIPS 感知损失（如果安装了 lpips 库则使用，否则回退到 VGG）
    """
    
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self._use_lpips = False
        try:
            import lpips
            self.model = lpips.LPIPS(net='vgg').to(device)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
            self._use_lpips = True
        except ImportError:
            self.model = VGGPerceptualLoss(device=device)
    
    def forward(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self._use_lpips:
            if rendered.dim() == 3:
                rendered = rendered.unsqueeze(0)
                target = target.unsqueeze(0)
            # LPIPS 期望输入在 [-1, 1] 范围
            r = rendered * 2 - 1
            t = target * 2 - 1
            return self.model(r, t).mean()
        else:
            self.model.to(rendered.device)
            return self.model(rendered, target)


class SSIMLoss(nn.Module):
    """
    SSIM 结构相似性损失
    
    衡量两张图像的结构相似性，比像素损失更符合人类视觉感知。
    返回 1 - SSIM，值越小表示越相似。
    """
    
    def __init__(self, window_size: int = 11, device: str = "cpu"):
        super().__init__()
        self.window_size = window_size
        self.device = device
        self._window = self._create_window(window_size)
    
    def _create_window(self, size: int) -> torch.Tensor:
        """创建高斯窗口"""
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * (size / 6.0) ** 2))
        g = torch.outer(g, g)
        g = g / g.sum()
        return g.unsqueeze(0).unsqueeze(0)
    
    def forward(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if rendered.dim() == 3:
            rendered = rendered.unsqueeze(0)
            target = target.unsqueeze(0)
        
        C = rendered.shape[1]
        window = self._window.expand(C, 1, -1, -1).to(rendered.device)
        
        # 计算均值
        mu_r = F.conv2d(rendered, window, padding=self.window_size // 2, groups=C)
        mu_t = F.conv2d(target, window, padding=self.window_size // 2, groups=C)
        
        mu_r_sq = mu_r ** 2
        mu_t_sq = mu_t ** 2
        mu_rt = mu_r * mu_t
        
        # 计算方差和协方差
        sigma_r_sq = F.conv2d(rendered ** 2, window, padding=self.window_size // 2, groups=C) - mu_r_sq
        sigma_t_sq = F.conv2d(target ** 2, window, padding=self.window_size // 2, groups=C) - mu_t_sq
        sigma_rt = F.conv2d(rendered * target, window, padding=self.window_size // 2, groups=C) - mu_rt
        
        # SSIM
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu_rt + C1) * (2 * sigma_rt + C2)) / \
                   ((mu_r_sq + mu_t_sq + C1) * (sigma_r_sq + sigma_t_sq + C2))
        
        return 1.0 - ssim_map.mean()


class ColorHistogramLoss(nn.Module):
    """
    颜色分布损失
    
    确保渲染图像的整体颜色分布与目标图像一致。
    使用软直方图（可微）计算颜色分布差异。
    """
    
    def __init__(self, num_bins: int = 64, device: str = "cpu"):
        super().__init__()
        self.num_bins = num_bins
        self.device = device
    
    def _soft_histogram(self, x: torch.Tensor, num_bins: int) -> torch.Tensor:
        """可微软直方图"""
        x = x.reshape(-1)
        bins = torch.linspace(0, 1, num_bins + 1, device=x.device)
        sigma = 1.0 / num_bins
        
        hist = torch.zeros(num_bins, device=x.device)
        for i in range(num_bins):
            center = (bins[i] + bins[i + 1]) / 2
            hist[i] = torch.exp(-0.5 * ((x - center) / sigma) ** 2).sum()
        
        hist = hist / (hist.sum() + 1e-8)
        return hist
    
    def forward(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if rendered.dim() == 4:
            rendered = rendered[0]
            target = target[0]
        
        loss = torch.tensor(0.0, device=rendered.device)
        for c in range(3):
            hist_r = self._soft_histogram(rendered[c], self.num_bins)
            hist_t = self._soft_histogram(target[c], self.num_bins)
            loss = loss + F.l1_loss(hist_r, hist_t)
        
        return loss / 3.0


class StrokeLengthLoss(nn.Module):
    """笔画长度正则化 - 防止优化出扭曲的毛线团笔画"""
    
    def __init__(self, weight: float = 0.01):
        super().__init__()
        self.weight = weight
    
    def forward(self, strokes: List[BrushStroke]) -> torch.Tensor:
        if not strokes:
            return torch.tensor(0.0)
        
        total = torch.tensor(0.0, device=strokes[0].control_points.device)
        for stroke in strokes:
            cp = stroke.control_points
            diffs = cp[1:] - cp[:-1]
            length = diffs.pow(2).sum(dim=1).sqrt().sum()
            total = total + length
        
        return self.weight * total / len(strokes)


class StrokeSmoothnessLoss(nn.Module):
    """笔画平滑度正则化 - 惩罚控制点的剧烈变化"""
    
    def __init__(self, weight: float = 0.005):
        super().__init__()
        self.weight = weight
    
    def forward(self, strokes: List[BrushStroke]) -> torch.Tensor:
        if not strokes:
            return torch.tensor(0.0)
        
        total = torch.tensor(0.0, device=strokes[0].control_points.device)
        for stroke in strokes:
            cp = stroke.control_points
            if cp.shape[0] >= 3:
                # 二阶差分（曲率）
                d1 = cp[1:] - cp[:-1]
                d2 = d1[1:] - d1[:-1]
                curvature = d2.pow(2).sum(dim=1).mean()
                total = total + curvature
        
        return self.weight * total / max(len(strokes), 1)


class CombinedLoss(nn.Module):
    """
    组合损失函数
    
    根据优化阶段动态调整各项损失的权重：
    - 底色阶段：重视像素对齐和颜色分布
    - 刻画阶段：平衡各项损失
    - 细节阶段：重视感知和结构
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
        self.perceptual_loss = LPIPSLoss(device=device)
        self.ssim_loss = SSIMLoss(device=device)
        self.color_loss = ColorHistogramLoss(device=device)
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
            rendered: [C, H, W] 渲染图像
            target: [C, H, W] 目标图像
            strokes: 笔画列表
            stage: 当前优化阶段
        """
        # Clamp rendered to prevent NaN
        rendered = rendered.clamp(0.0, 1.0)

        # NaN protection
        if torch.isnan(rendered).any() or torch.isinf(rendered).any():
            fake = rendered.sum() * 0
            return fake, {"pixel": 1e3, "perceptual": 1e3, "ssim": 1e3,
                           "color_hist": 0.0, "length": 0.0, "smooth": 0.0, "total": 1e3}

        l_pixel = self.pixel_loss(rendered, target)
        l_perceptual = self.perceptual_loss(rendered, target)
        l_ssim = self.ssim_loss(rendered, target)
        l_color = self.color_loss(rendered, target)
        l_length = self.length_loss(strokes)
        l_smooth = self.smoothness_loss(strokes)
        
        # 根据阶段动态调整权重
        if stage == 0:
            w_pixel, w_perceptual, w_ssim, w_color = 2.0, 5.0, 0.5, 2.0
        elif stage == 1:
            w_pixel, w_perceptual, w_ssim, w_color = 1.0, 10.0, 2.0, 1.0
        else:
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
