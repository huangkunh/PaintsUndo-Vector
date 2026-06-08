"""
注意力引导的笔画放置

使用视觉注意力机制引导笔画放置位置：
1. 显著性检测：找到视觉上最显著的区域
2. 残差注意力：在当前渲染与目标的差异最大的区域放置笔画
3. 边缘注意力：在边缘区域放置更多笔画
4. 颜色注意力：在颜色差异最大的区域放置笔画

这些注意力机制确保笔画被放置在最需要的位置，
而不是随机分布，从而更高效地逼近目标图像。
"""

from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from brushes.base import BrushStroke


class AttentionMap:
    """
    注意力图计算器
    
    计算多种注意力图，用于引导笔画放置：
    - 残差注意力：|rendered - target| 的空间分布
    - 边缘注意力：目标图像的边缘强度分布
    - 显著性注意力：视觉显著性分布
    - 颜色注意力：颜色差异的空间分布
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
    
    def compute_residual_attention(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算残差注意力图。
        
        在当前渲染与目标差异最大的区域，
        放置更多笔画以减少差异。
        
        Args:
            rendered: [C, H, W] 当前渲染图像
            target: [C, H, W] 目标图像
            
        Returns:
            [H, W] 注意力图，值越大表示越需要笔画
        """
        diff = (rendered - target).abs().mean(dim=0)  # [H, W]
        
        # 归一化到 [0, 1]
        if diff.max() > 0:
            attention = diff / diff.max()
        else:
            attention = diff
        
        return attention
    
    def compute_edge_attention(
        self,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算边缘注意力图。
        
        在边缘区域放置更多笔画，
        模拟画家在轮廓处更加仔细的绘画习惯。
        
        使用 Sobel 算子检测边缘。
        """
        # 转为灰度
        gray = target.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, H, W]
        
        # Sobel 算子
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32, device=self.device
        ).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32, device=self.device
        ).view(1, 1, 3, 3)
        
        edge_x = F.conv2d(gray, sobel_x, padding=1)
        edge_y = F.conv2d(gray, sobel_y, padding=1)
        
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2).squeeze()  # [H, W]
        
        # 归一化
        if edge.max() > 0:
            attention = edge / edge.max()
        else:
            attention = edge
        
        return attention
    
    def compute_saliency_attention(
        self,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算显著性注意力图。
        
        使用简化的显著性检测：
        基于颜色对比度和中心偏好。
        """
        C, H, W = target.shape
        
        # 计算颜色对比度
        mean_color = target.mean(dim=[1, 2], keepdim=True)
        color_diff = (target - mean_color).abs().mean(dim=0)  # [H, W]
        
        # 中心偏好（高斯权重）
        y_coords = torch.linspace(-1, 1, H, device=self.device)
        x_coords = torch.linspace(-1, 1, W, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        center_weight = torch.exp(-(grid_x ** 2 + grid_y ** 2) / 0.5)
        
        # 组合
        saliency = color_diff * center_weight
        
        # 归一化
        if saliency.max() > 0:
            attention = saliency / saliency.max()
        else:
            attention = saliency
        
        return attention
    
    def compute_color_attention(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算颜色注意力图。
        
        在颜色差异最大的区域放置笔画。
        使用 LAB 颜色空间的差异。
        """
        # 简化：使用 RGB 空间的加权差异
        # 人类视觉对绿色更敏感
        weights = torch.tensor([0.299, 0.587, 0.114], device=self.device).view(3, 1, 1)
        diff = ((rendered - target) * weights).abs().sum(dim=0)  # [H, W]
        
        # 归一化
        if diff.max() > 0:
            attention = diff / diff.max()
        else:
            attention = diff
        
        return attention
    
    def compute_combined_attention(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        stage: int = 0,
    ) -> torch.Tensor:
        """
        计算组合注意力图。
        
        根据不同阶段调整各注意力图的权重：
        - 底色阶段：颜色注意力最重要
        - 刻画阶段：残差注意力最重要
        - 细节阶段：边缘注意力最重要
        
        Args:
            rendered: [C, H, W] 当前渲染图像
            target: [C, H, W] 目标图像
            stage: 当前阶段索引
            
        Returns:
            [H, W] 组合注意力图
        """
        residual = self.compute_residual_attention(rendered, target)
        edge = self.compute_edge_attention(target)
        saliency = self.compute_saliency_attention(target)
        color = self.compute_color_attention(rendered, target)
        
        if stage == 0:
            # 底色阶段：颜色和显著性最重要
            w_r, w_e, w_s, w_c = 0.2, 0.1, 0.3, 0.4
        elif stage == 1:
            # 刻画阶段：残差和边缘最重要
            w_r, w_e, w_s, w_c = 0.4, 0.3, 0.1, 0.2
        else:
            # 细节阶段：边缘和残差最重要
            w_r, w_e, w_s, w_c = 0.3, 0.4, 0.1, 0.2
        
        combined = w_r * residual + w_e * edge + w_s * saliency + w_c * color
        
        # 归一化
        if combined.max() > 0:
            combined = combined / combined.max()
        
        return combined
    
    def sample_stroke_position(
        self,
        attention_map: torch.Tensor,
        num_samples: int = 1,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        根据注意力图采样笔画位置。
        
        注意力值越高的区域被采样的概率越大，
        但也保留一定的随机性（由 temperature 控制）。
        
        Args:
            attention_map: [H, W] 注意力图
            num_samples: 采样数量
            temperature: 温度参数（越高越均匀，越低越集中在高注意力区域）
            
        Returns:
            [num_samples, 2] 采样位置（归一化坐标 [0, 1]）
        """
        H, W = attention_map.shape
        
        # 展平并应用温度
        flat_attention = attention_map.reshape(-1)
        flat_attention = (flat_attention / max(temperature, 0.01)).exp()
        
        # 归一化为概率分布
        probs = flat_attention / flat_attention.sum()
        
        # 采样
        indices = torch.multinomial(probs, num_samples, replacement=True)
        
        # 转换为坐标
        y_coords = indices // W
        x_coords = indices % W
        
        positions = torch.stack([
            x_coords.float() / W,
            y_coords.float() / H,
        ], dim=1)
        
        return positions
    
    def sample_stroke_direction(
        self,
        position: torch.Tensor,
        target: torch.Tensor,
    ) -> float:
        """
        根据目标图像的局部梯度方向确定笔画方向。
        
        模拟画家沿边缘方向或垂直于边缘方向绘画的习惯。
        
        Args:
            position: [2] 笔画位置（归一化坐标）
            target: [C, H, W] 目标图像
            
        Returns:
            笔画方向角度（弧度）
        """
        C, H, W = target.shape
        
        # 获取位置附近的梯度
        px = int(position[0].item() * (W - 1))
        py = int(position[1].item() * (H - 1))
        
        # 计算梯度
        gray = target.mean(dim=0)
        
        # 有限差分
        dx = torch.zeros_like(gray)
        dy = torch.zeros_like(gray)
        
        if px > 0 and px < W - 1:
            dx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        if py > 0 and py < H - 1:
            dy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        
        # 局部梯度方向
        local_dx = dx[py, px].item() if 0 <= py < H and 0 <= px < W else 0
        local_dy = dy[py, px].item() if 0 <= py < H and 0 <= px < W else 0
        
        # 笔画方向垂直于梯度方向（沿边缘方向）
        angle = np.arctan2(-local_dx, local_dy)
        
        # 添加少量随机偏移
        angle += np.random.randn() * 0.3
        
        return angle
