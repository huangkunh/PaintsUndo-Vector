"""
笔画初始化策略

根据目标图像的特征，智能初始化笔画参数。
不同阶段使用不同的初始化策略：
- 底色阶段：基于图像色块分析
- 刻画阶段：基于边缘检测
- 细节阶段：基于高频信息提取
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

from brushes.base import BrushStroke


def image_to_tensor(image_path: str, size: Tuple[int, int], device: str = "cpu") -> torch.Tensor:
    """加载图像并转换为张量 [C, H, W]，值域 [0, 1]"""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size[0], size[1]), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).to(device)
    return tensor


def extract_color_palette(
    image: torch.Tensor,
    num_colors: int = 8,
) -> torch.Tensor:
    """
    从图像中提取主要颜色。
    
    使用 K-Means 聚类提取主要颜色。
    
    Args:
        image: [C, H, W] 图像张量
        num_colors: 提取的颜色数量
        
    Returns:
        [num_colors, 3] 颜色张量
    """
    # 转换为 numpy
    img_np = image.permute(1, 2, 0).cpu().numpy()
    pixels = img_np.reshape(-1, 3).astype(np.float32)
    
    # K-Means 聚类
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels, num_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    
    centers = torch.from_numpy(centers).to(image.device)
    return centers


def extract_edges(
    image: torch.Tensor,
    low_threshold: int = 50,
    high_threshold: int = 150,
) -> torch.Tensor:
    """
    从图像中提取边缘。
    
    使用 Canny 边缘检测。
    
    Args:
        image: [C, H, W] 图像张量
        
    Returns:
        [H, W] 边缘掩码
    """
    img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return torch.from_numpy(edges.astype(np.float32) / 255.0).to(image.device)


def extract_high_frequency(
    image: torch.Tensor,
    blur_kernel: int = 15,
) -> torch.Tensor:
    """
    提取图像的高频信息（细节）。
    
    通过原图减去模糊图得到高频分量。
    """
    img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(img_np, (blur_kernel, blur_kernel), 0)
    high_freq = cv2.absdiff(img_np, blurred)
    high_freq = high_freq.astype(np.float32) / 255.0
    return torch.from_numpy(high_freq).permute(2, 0, 1).to(image.device)


def initialize_strokes_stage1(
    target_image: torch.Tensor,
    num_strokes: int = 20,
    max_width: float = 50.0,
    min_width: float = 20.0,
    num_control_points: int = 5,
    canvas_size: Tuple[int, int] = (640, 480),
    device: str = "cpu",
) -> Tuple[List[BrushStroke], List[str]]:
    """
    阶段1初始化：铺底色
    
    基于图像色块分析，初始化粗大的笔画覆盖大面积色彩。
    使用马克笔和喷笔笔刷。
    
    Args:
        target_image: 目标图像 [C, H, W]
        num_strokes: 笔画数量
        max_width: 最大笔画宽度
        min_width: 最小笔画宽度
        num_control_points: 每条笔画的控制点数
        
    Returns:
        strokes: 笔画列表
        brush_names: 对应的笔刷名称列表
    """
    # 提取主要颜色
    palette = extract_color_palette(target_image, num_colors=min(num_strokes, 8))
    
    strokes = []
    brush_names = []
    
    H, W = target_image.shape[1], target_image.shape[2]
    
    for i in range(num_strokes):
        # 选择颜色
        color_idx = i % palette.shape[0]
        color_rgb = palette[color_idx]
        color = torch.cat([color_rgb, torch.tensor([1.0], device=device)])
        
        # 随机初始化笔画宽度
        width = torch.rand(1, device=device) * (max_width - min_width) + min_width
        
        # 随机初始化控制点（在画布范围内）
        # 使用均匀分布覆盖整个画布
        raw_points = torch.rand(num_control_points, 2, device=device)
        
        stroke = BrushStroke(
            num_control_points=num_control_points,
            canvas_size=canvas_size,
            init_width=width.item(),
            init_color=color,
            init_opacity=0.8,
            device=device,
        )
        # 覆盖随机初始化的控制点
        stroke.raw_control_points = nn.Parameter(raw_points * 0.2 - 0.1)
        
        strokes.append(stroke)
        
        # 交替使用马克笔和喷笔
        brush_names.append("marker" if i % 3 != 0 else "airbrush")
    
    return strokes, brush_names


def initialize_strokes_stage2(
    target_image: torch.Tensor,
    num_strokes: int = 100,
    max_width: float = 15.0,
    min_width: float = 5.0,
    num_control_points: int = 7,
    canvas_size: Tuple[int, int] = (640, 480),
    device: str = "cpu",
) -> Tuple[List[BrushStroke], List[str]]:
    """
    阶段2初始化：形体刻画
    
    基于边缘检测，初始化中等粗细的笔画勾勒轮廓和色块过渡。
    使用压感笔和水彩笔刷。
    """
    # 提取边缘
    edges = extract_edges(target_image)
    
    # 在边缘附近采样笔画起点
    edge_points = torch.nonzero(edges > 0.3, as_tuple=False)
    
    strokes = []
    brush_names = []
    
    H, W = target_image.shape[1], target_image.shape[2]
    
    for i in range(num_strokes):
        # 从边缘点或随机位置初始化
        if edge_points.shape[0] > 0 and i < edge_points.shape[0]:
            idx = torch.randint(0, edge_points.shape[0], (1,)).item()
            start_y, start_x = edge_points[idx]
            start_x = start_x.float() / W
            start_y = start_y.float() / H
        else:
            start_x = torch.rand(1, device=device).item()
            start_y = torch.rand(1, device=device).item()
        
        # 采样目标图像在该位置的颜色
        px = int(start_x * (W - 1))
        py = int(start_y * (H - 1))
        px = max(0, min(W - 1, px))
        py = max(0, min(H - 1, py))
        color_rgb = target_image[:, py, px]
        color = torch.cat([color_rgb, torch.tensor([1.0], device=device)])
        
        # 初始化控制点 - 从起点出发的曲线
        width = torch.rand(1, device=device) * (max_width - min_width) + min_width
        
        raw_points = torch.zeros(num_control_points, 2, device=device)
        raw_points[0] = torch.tensor([start_x, start_y]) * 2 - 1
        
        # 沿随机方向延伸
        angle = torch.rand(1).item() * 2 * np.pi
        length = torch.rand(1).item() * 0.3 + 0.1
        for j in range(1, num_control_points):
            t = j / (num_control_points - 1)
            offset_x = start_x + length * t * np.cos(angle) + torch.randn(1).item() * 0.05
            offset_y = start_y + length * t * np.sin(angle) + torch.randn(1).item() * 0.05
            raw_points[j] = torch.tensor([offset_x, offset_y]) * 2 - 1
        
        stroke = BrushStroke(
            num_control_points=num_control_points,
            canvas_size=canvas_size,
            init_width=width.item(),
            init_color=color,
            init_opacity=0.7,
            device=device,
        )
        stroke.raw_control_points = nn.Parameter(raw_points)
        
        strokes.append(stroke)
        
        # 交替使用压感笔和水彩笔
        brush_names.append("pressure" if i % 2 == 0 else "watercolor")
    
    return strokes, brush_names


def initialize_strokes_stage3(
    target_image: torch.Tensor,
    num_strokes: int = 300,
    max_width: float = 3.0,
    min_width: float = 1.0,
    num_control_points: int = 5,
    canvas_size: Tuple[int, int] = (640, 480),
    device: str = "cpu",
) -> Tuple[List[BrushStroke], List[str]]:
    """
    阶段3初始化：细节线稿
    
    基于高频信息提取，初始化细小的笔画勾勒细节。
    使用铅笔和排线笔刷。
    """
    # 提取高频信息
    high_freq = extract_high_frequency(target_image)
    high_freq_gray = high_freq.mean(dim=0)
    
    # 在高频区域采样笔画起点
    detail_points = torch.nonzero(high_freq_gray > 0.1, as_tuple=False)
    
    strokes = []
    brush_names = []
    
    H, W = target_image.shape[1], target_image.shape[2]
    
    for i in range(num_strokes):
        # 从高频区域或随机位置初始化
        if detail_points.shape[0] > 0 and i < detail_points.shape[0]:
            idx = torch.randint(0, detail_points.shape[0], (1,)).item()
            start_y, start_x = detail_points[idx]
            start_x = start_x.float() / W
            start_y = start_y.float() / H
        else:
            start_x = torch.rand(1, device=device).item()
            start_y = torch.rand(1, device=device).item()
        
        # 采样颜色
        px = int(start_x * (W - 1))
        py = int(start_y * (H - 1))
        px = max(0, min(W - 1, px))
        py = max(0, min(H - 1, py))
        color_rgb = target_image[:, py, px]
        color = torch.cat([color_rgb, torch.tensor([1.0], device=device)])
        
        width = torch.rand(1, device=device) * (max_width - min_width) + min_width
        
        raw_points = torch.zeros(num_control_points, 2, device=device)
        raw_points[0] = torch.tensor([start_x, start_y]) * 2 - 1
        
        # 短距离延伸
        angle = torch.rand(1).item() * 2 * np.pi
        length = torch.rand(1).item() * 0.1 + 0.02
        for j in range(1, num_control_points):
            t = j / (num_control_points - 1)
            offset_x = start_x + length * t * np.cos(angle) + torch.randn(1).item() * 0.02
            offset_y = start_y + length * t * np.sin(angle) + torch.randn(1).item() * 0.02
            raw_points[j] = torch.tensor([offset_x, offset_y]) * 2 - 1
        
        stroke = BrushStroke(
            num_control_points=num_control_points,
            canvas_size=canvas_size,
            init_width=width.item(),
            init_color=color,
            init_opacity=0.9,
            device=device,
        )
        stroke.raw_control_points = nn.Parameter(raw_points)
        
        strokes.append(stroke)
        
        # 交替使用铅笔和排线
        brush_names.append("pencil" if i % 3 != 0 else "hatching")
    
    return strokes, brush_names
