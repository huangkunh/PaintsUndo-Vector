"""
损失函数模块

实现多种损失函数用于笔画优化：
- L_pixel: L1 像素损失
- L_perceptual: LPIPS 感知损失
- L_length: 笔画长度正则化
- L_smooth: 笔画平滑度正则化
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from brushes.base import BrushStroke


class PixelLoss(nn.Module):
    """
    L1/L2 像素损失
    
    计算渲染图像与目标图像之间的像素级差异。
    L1 损失对颜色偏差更敏感，L2 损失对大偏差更敏感。
    """
    
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
        """
        Args:
            rendered: 渲染图像 [B, C, H, W] 或 [C, H, W]
            target: 目标图像 [B, C, H, W] 或 [C, H, W]
        """
        if rendered.dim() == 3:
            rendered = rendered.unsqueeze(0)
            target = target.unsqueeze(0)
        return self.criterion(rendered, target)


class PerceptualLoss(nn.Module):
    """
    LPIPS 感知损失
    
    使用预训练的 VGG 网络提取特征，计算特征空间的差异。
    比像素损失更符合人类视觉感知。
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
            print("Warning: lpips not installed, perceptual loss will be replaced by L2 loss")
            self._available = False
    
    def forward(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rendered: 渲染图像 [C, H, W]，值域 [0, 1]
            target: 目标图像 [C, H, W]，值域 [0, 1]
        """
        if rendered.dim() == 3:
            rendered = rendered.unsqueeze(0)
            target = target.unsqueeze(0)
        
        # LPIPS 期望输入在 [-1, 1] 范围
        rendered_norm = rendered * 2 - 1
        target_norm = target * 2 - 1
        
        if self._available:
            return self.model(rendered_norm, target_norm).mean()
        else:
            # 回退到 L2 损失
            return F.mse_loss(rendered, target)


class StrokeLengthLoss(nn.Module):
    """
    笔画长度正则化损失
    
    惩罚过长的笔画，防止优化出扭曲的毛线团笔画。
    对控制点之间的距离进行 L2 惩罚。
    """
    
    def __init__(self, weight: float = 0.01):
        super().__init__()
        self.weight = weight
    
    def forward(self, strokes: List[BrushStroke]) -> torch.Tensor:
        """
        计算所有笔画的总长度损失。
        """
        total_loss = torch.tensor(0.0, device=strokes[0].device if strokes else "cpu")
        
        for stroke in strokes:
            cp = stroke.control_points  # [N, 2]
            # 相邻控制点之间的距离
            diffs = cp[1:] - cp[:-1]
            distances = torch.sqrt((diffs ** 2).sum(dim=1) + 1e-8)
            total_loss = total_loss + distances.sum()
        
        return total_loss * self.weight


class StrokeSmoothnessLoss(nn.Module):
    """
    笔画平滑度正则化损失
    
    惩罚笔画中的急转弯，鼓励平滑的曲线。
    通过计算相邻线段的角度变化来衡量。
    """
    
    def __init__(self, weight: float = 0.005):
        super().__init__()
        self.weight = weight
    
    def forward(self, strokes: List[BrushStroke]) -> torch.Tensor:
        """
        计算所有笔画的平滑度损失。
        """
        total_loss = torch.tensor(0.0, device=strokes[0].device if strokes else "cpu")
        
        for stroke in strokes:
            cp = stroke.control_points  # [N, 2]
            if cp.shape[0] < 3:
                continue
            
            # 计算一阶差分（方向向量）
            d1 = cp[1:] - cp[:-1]  # [N-1, 2]
            
            # 计算二阶差分（方向变化）
            d2 = d1[1:] - d1[:-1]  # [N-2, 2]
            
            # 方向变化的幅度
            curvature = (d2 ** 2).sum(dim=1)
            total_loss = total_loss + curvature.sum()
        
        return total_loss * self.weight


class CombinedLoss(nn.Module):
    """
    组合损失函数
    
    L_total = λ_pixel * L_pixel + λ_perceptual * L_perceptual 
            + λ_length * L_length + λ_smooth * L_smooth
    """
    
    def __init__(
        self,
        lambda_pixel: float = 1.0,
        lambda_perceptual: float = 10.0,
        lambda_length: float = 0.01,
        lambda_smooth: float = 0.005,
        device: str = "cpu",
    ):
        super().__init__()
        self.lambda_pixel = lambda_pixel
        self.lambda_perceptual = lambda_perceptual
        self.lambda_length = lambda_length
        self.lambda_smooth = lambda_smooth
        
        self.pixel_loss = PixelLoss(loss_type="l1")
        self.perceptual_loss = PerceptualLoss(device=device)
        self.length_loss = StrokeLengthLoss(weight=lambda_length)
        self.smoothness_loss = StrokeSmoothnessLoss(weight=lambda_smooth)
    
    def forward(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        strokes: List[BrushStroke],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算组合损失。
        
        Args:
            rendered: 渲染图像 [C, H, W]
            target: 目标图像 [C, H, W]
            strokes: 笔画列表
            
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的值
        """
        l_pixel = self.pixel_loss(rendered, target)
        l_perceptual = self.perceptual_loss(rendered, target)
        l_length = self.length_loss(strokes)
        l_smooth = self.smoothness_loss(strokes)
        
        total = (
            self.lambda_pixel * l_pixel
            + self.lambda_perceptual * l_perceptual
            + l_length
            + l_smooth
        )
        
        loss_dict = {
            "pixel": l_pixel.item(),
            "perceptual": l_perceptual.item(),
            "length": l_length.item(),
            "smooth": l_smooth.item(),
            "total": total.item(),
        }
        
        return total, loss_dict
