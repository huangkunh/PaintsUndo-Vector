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
    
    计算多种注意力图，用于引导笔画放置。
    所有计算均为纯 PyTorch 实现，无需 cv2 依赖。
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
        
        在当前渲染与目标差异最大的区域，放置更多笔画。
        """
        diff = (rendered - target).abs().mean(dim=0)
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
        计算边缘注意力图（纯 PyTorch Sobel 算子）。
        
        在边缘区域放置更多笔画，模拟画家在轮廓处更加仔细的绘画习惯。
        """
        gray = target.mean(dim=0, keepdim=True).unsqueeze(0)
        
        # Sobel 算子
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32, device=self.device
        ).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32, device=self.device
        ).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        
        edge = (grad_x ** 2 + grad_y ** 2).sqrt().squeeze(0, 1)
        
        if edge.max() > 0:
            edge = edge / edge.max()
        
        return edge
    
    def compute_saliency_attention(
        self,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算显著性注意力图。
        
        使用中心-周围差异模型计算视觉显著性，
        模拟人类视觉系统的注意力机制。
        """
        gray = target.mean(dim=0, keepdim=True).unsqueeze(0)
        
        # 多尺度中心-周围差异
        saliency = torch.zeros_like(gray.squeeze(0).squeeze(0))
        
        for sigma in [4, 8, 16]:
            kernel_size = int(sigma * 6) | 1  # 确保奇数
            kernel = _gaussian_kernel_1d(kernel_size, sigma, self.device)
            
            # 中心（细尺度）
            center = _separable_conv2d(gray, kernel)
            # 周围（粗尺度）
            surround = _separable_conv2d(center, kernel)
            
            diff = (center - surround).abs().squeeze(0).squeeze(0)
            if diff.max() > 0:
                diff = diff / diff.max()
            saliency += diff
        
        saliency = saliency / 3.0
        
        if saliency.max() > 0:
            saliency = saliency / saliency.max()
        
        return saliency
    
    def compute_color_attention(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算颜色注意力图。
        
        在颜色差异最大的区域放置笔画。
        使用 LAB 颜色空间的感知差异。
        """
        # 简化的感知颜色差异（RGB 空间加权）
        r_diff = (rendered[0] - target[0]).abs()
        g_diff = (rendered[1] - target[1]).abs()
        b_diff = (rendered[2] - target[2]).abs()
        
        # 人眼对绿色更敏感
        color_diff = 0.3 * r_diff + 0.59 * g_diff + 0.11 * b_diff
        
        if color_diff.max() > 0:
            color_diff = color_diff / color_diff.max()
        
        return color_diff
    
    def compute_combined_attention(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        residual_weight: float = 0.4,
        edge_weight: float = 0.3,
        saliency_weight: float = 0.1,
        color_weight: float = 0.2,
    ) -> torch.Tensor:
        """
        计算组合注意力图。
        
        将多种注意力图加权融合，得到最终的笔画放置引导图。
        """
        residual = self.compute_residual_attention(rendered, target)
        edge = self.compute_edge_attention(target)
        saliency = self.compute_saliency_attention(target)
        color = self.compute_color_attention(rendered, target)
        
        # 统一尺寸
        target_size = residual.shape
        edge = F.interpolate(
            edge.unsqueeze(0).unsqueeze(0), size=target_size, mode='bilinear', align_corners=False
        ).squeeze(0).squeeze(0)
        saliency = F.interpolate(
            saliency.unsqueeze(0).unsqueeze(0), size=target_size, mode='bilinear', align_corners=False
        ).squeeze(0).squeeze(0)
        color = F.interpolate(
            color.unsqueeze(0).unsqueeze(0), size=target_size, mode='bilinear', align_corners=False
        ).squeeze(0).squeeze(0)
        
        combined = (
            residual_weight * residual
            + edge_weight * edge
            + saliency_weight * saliency
            + color_weight * color
        )
        
        if combined.max() > 0:
            combined = combined / combined.max()
        
        return combined
    
    def sample_positions_from_attention(
        self,
        attention_map: torch.Tensor,
        num_samples: int = 10,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        从注意力图中采样笔画放置位置。
        
        使用 softmax 温度控制采样分布的尖锐程度：
        - temperature → 0: 趋向贪心采样（只选最大值）
        - temperature → ∞: 趋向均匀采样
        """
        H, W = attention_map.shape
        
        # 展平并应用温度
        flat = attention_map.reshape(-1)
        probs = F.softmax(flat / max(temperature, 0.01), dim=0)
        
        # 采样
        indices = torch.multinomial(probs, num_samples, replacement=True)
        
        y_coords = indices // W
        x_coords = indices % W
        
        positions = torch.stack([
            x_coords.float() / W,
            y_coords.float() / H,
        ], dim=1)
        
        return positions


def sample_from_attention_map(
    attention_map: torch.Tensor,
    num_samples: int = 10,
    temperature: float = 1.0,
) -> torch.Tensor:
    """从注意力图采样位置的便捷函数"""
    device = attention_map.device
    am = AttentionMap(device=device)
    return am.sample_positions_from_attention(attention_map, num_samples, temperature)


def compute_local_gradient_direction(
    target_image: torch.Tensor,
    position: Tuple[float, float],
) -> float:
    """
    根据目标图像的局部梯度方向确定笔画方向。
    
    模拟画家沿边缘方向或垂直于边缘方向绘画的习惯。
    """
    C, H, W = target_image.shape
    px = int(position[0] * (W - 1))
    py = int(position[1] * (H - 1))
    
    gray = target_image.mean(dim=0)
    
    dx = torch.zeros_like(gray)
    dy = torch.zeros_like(gray)
    
    if W > 2:
        dx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    if H > 2:
        dy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    
    local_dx = dx[py, px].item() if 0 <= py < H and 0 <= px < W else 0
    local_dy = dy[py, px].item() if 0 <= py < H and 0 <= px < W else 0
    
    # 笔画方向垂直于梯度方向（沿边缘方向）
    angle = np.arctan2(-local_dx, local_dy)
    angle += np.random.randn() * 0.3
    
    return angle


# ==================== 辅助函数 ====================

def _gaussian_kernel_1d(size: int, sigma: float, device: str = "cpu") -> torch.Tensor:
    """创建 1D 高斯卷积核"""
    x = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel


def _separable_conv2d(x: torch.Tensor, kernel_1d: torch.Tensor) -> torch.Tensor:
    """使用分离卷积进行 2D 高斯模糊"""
    C = x.shape[1]
    k = kernel_1d.shape[0]
    
    # 水平卷积核
    kh = kernel_1d.view(1, 1, 1, k).expand(C, 1, 1, k)
    # 垂直卷积核
    kv = kernel_1d.view(1, 1, k, 1).expand(C, 1, k, 1)
    
    padding_h = k // 2
    padding_v = k // 2
    
    out = F.conv2d(x, kh, padding=(0, padding_h), groups=C)
    out = F.conv2d(out, kv, padding=(padding_v, 0), groups=C)
    
    return out
