"""
笔画初始化策略 - 基于图像分析的智能初始化（纯 PyTorch 实现）

根据目标图像的特征，智能初始化笔画参数。
不同阶段使用不同的初始化策略：
- 底色阶段：基于颜色聚类的大面积覆盖
- 刻画阶段：基于边缘检测和显著性区域的笔画放置
- 细节阶段：基于高频信息和高梯度区域的精细笔画

关键改进：
1. 纯 PyTorch 实现，无需 cv2 依赖
2. 颜色聚类初始化：从目标图像提取主色调，按颜色区域放置笔画
3. 边缘引导初始化：沿边缘方向放置笔画，模拟画家的轮廓勾勒
4. 显著性引导：在视觉显著区域放置更多笔画
5. 渐进式细化：每个阶段在前一阶段的残差上初始化
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from brushes.base import BrushStroke
from utils.image import resize_image


def image_to_tensor(image_path: str, size: Tuple[int, int], device: str = "cpu") -> torch.Tensor:
    """加载图像并转换为张量 [C, H, W]，值域 [0, 1]"""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size[0], size[1]), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).to(device)
    return tensor


def pil_to_tensor(image, size: Tuple[int, int], device: str = "cpu") -> torch.Tensor:
    """将 PIL Image 转换为张量"""
    from PIL import Image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    image = image.resize((size[0], size[1]), Image.LANCZOS)
    arr = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).to(device)


def extract_color_palette(
    image: torch.Tensor,
    num_colors: int = 8,
) -> torch.Tensor:
    """
    从图像中提取主要颜色（纯 PyTorch K-Means 聚类）。
    
    模拟画家调色板：提取图像的主要色调，
    用于初始化底色阶段的笔画颜色。
    """
    C, H, W = image.shape
    pixels = image.permute(1, 2, 0).reshape(-1, 3)  # [H*W, 3]
    
    # 如果像素太多，随机采样
    if pixels.shape[0] > 10000:
        indices = torch.randperm(pixels.shape[0])[:10000]
        pixels = pixels[indices]
    
    # 简单的 K-Means 实现
    device = image.device
    
    # 随机初始化聚类中心
    perm = torch.randperm(pixels.shape[0])[:num_colors]
    centers = pixels[perm].clone()
    
    for _ in range(20):  # 20 次 K-Means 迭代
        # 分配每个像素到最近的中心
        dists = torch.cdist(pixels, centers)  # [N, K]
        labels = dists.argmin(dim=1)  # [N]
        
        # 更新中心
        for k in range(num_colors):
            mask = labels == k
            if mask.sum() > 0:
                centers[k] = pixels[mask].mean(dim=0)
    
    # 按颜色区域大小排序
    label_counts = torch.bincount(labels, minlength=num_colors)
    sorted_indices = label_counts.argsort(descending=True)
    centers = centers[sorted_indices]
    
    return centers


def extract_edges_torch(
    image: torch.Tensor,
    threshold: float = 0.1,
) -> torch.Tensor:
    """
    使用 Sobel 算子提取边缘（纯 PyTorch 实现）。
    
    Args:
        image: [C, H, W] 图像张量
        threshold: 边缘强度阈值
        
    Returns:
        [H, W] 边缘强度图
    """
    gray = image.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, H, W]
    
    # Sobel 算子
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
    
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    
    edges = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8).squeeze()
    
    # 归一化
    if edges.max() > 0:
        edges = edges / edges.max()
    
    return edges


def extract_saliency(
    image: torch.Tensor,
) -> torch.Tensor:
    """
    简单的显著性检测（纯 PyTorch 实现）。
    
    基于颜色对比度的显著性：与平均颜色差异越大的区域越显著。
    """
    C, H, W = image.shape
    
    # 计算平均颜色
    mean_color = image.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
    
    # 颜色差异
    diff = (image - mean_color).pow(2).sum(dim=0).sqrt()  # [H, W]
    
    # 归一化
    if diff.max() > 0:
        diff = diff / diff.max()
    
    # 高斯平滑
    kernel_size = 15
    sigma = kernel_size / 6.0
    x = torch.arange(kernel_size, device=image.device) - kernel_size // 2
    gauss = torch.exp(-x.float() ** 2 / (2 * sigma ** 2))
    kernel_1d = gauss / gauss.sum()
    kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
    
    diff = diff.unsqueeze(0).unsqueeze(0)
    padding = kernel_size // 2
    diff = F.conv2d(diff, kernel_2d, padding=padding).squeeze()
    
    return diff


def extract_high_frequency(
    image: torch.Tensor,
) -> torch.Tensor:
    """
    提取高频信息（纯 PyTorch 实现）。
    
    高频信息对应图像的细节和纹理，
    用于引导细节阶段的笔画放置。
    """
    C, H, W = image.shape
    
    # 低通滤波
    kernel_size = 9
    sigma = kernel_size / 4.0
    x = torch.arange(kernel_size, device=image.device) - kernel_size // 2
    gauss = torch.exp(-x.float() ** 2 / (2 * sigma ** 2))
    kernel_1d = gauss / gauss.sum()
    kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)
    
    img = image.unsqueeze(0)
    padding = kernel_size // 2
    low_pass = F.conv2d(img, kernel_2d, padding=padding, groups=C)
    
    # 高频 = 原图 - 低通
    high_freq = (img - low_pass).abs().squeeze(0).mean(dim=0)
    
    if high_freq.max() > 0:
        high_freq = high_freq / high_freq.max()
    
    return high_freq


def sample_from_attention_map(
    attention_map: torch.Tensor,
    num_samples: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    从注意力图中采样位置。
    
    值越大的区域被采样的概率越高，
    模拟画家在重要区域放置更多笔画的习惯。
    """
    H, W = attention_map.shape
    
    # 温度缩放
    probs = (attention_map.reshape(-1) / max(temperature, 0.01)).softmax(dim=0)
    
    # 采样
    indices = torch.multinomial(probs, num_samples, replacement=True)
    
    y_coords = indices // W
    x_coords = indices % W
    
    positions = torch.stack([
        x_coords.float() / W,
        y_coords.float() / H,
    ], dim=1)
    
    return positions


def compute_local_gradient_direction(
    image: torch.Tensor,
    position: Tuple[float, float],
) -> float:
    """
    计算目标图像在指定位置的局部梯度方向。
    
    模拟画家沿边缘方向或垂直于边缘方向绘画的习惯。
    """
    C, H, W = image.shape
    px = int(position[0] * (W - 1))
    py = int(position[1] * (H - 1))
    
    gray = image.mean(dim=0)
    
    # Sobel 梯度
    if 1 <= py < H - 1 and 1 <= px < W - 1:
        dx = gray[py, px + 1] - gray[py, px - 1]
        dy = gray[py + 1, px] - gray[py - 1, px]
    else:
        dx, dy = 0.0, 0.0
    
    # 笔画方向垂直于梯度方向（沿边缘方向）
    angle = float(np.arctan2(-float(dx), float(dy)))
    angle += np.random.randn() * 0.3  # 少量随机偏移
    
    return angle


# ===================== 阶段初始化函数 =====================


def initialize_strokes_stage1(
    target_image: torch.Tensor,
    canvas_size: Tuple[int, int] = (640, 480),
    num_strokes: int = 20,
    max_width: float = 50.0,
    min_width: float = 20.0,
    num_control_points: int = 5,
    device: str = "cpu",
    rendered_image: Optional[torch.Tensor] = None,
) -> Tuple[List[BrushStroke], List[str]]:
    """
    Stage 1: 铺底色初始化
    
    基于颜色聚类：从目标图像提取主色调，
    在每个颜色区域的中心放置粗笔画。
    
    模拟画家用大号画笔铺底色的过程。
    """
    # 提取颜色调色板
    palette = extract_color_palette(target_image, num_colors=min(num_strokes, 16))
    
    # 计算显著性图
    saliency = extract_saliency(target_image)
    
    strokes = []
    brush_names = []
    
    for i in range(num_strokes):
        # 选择颜色
        color_idx = i % len(palette)
        color_rgb = palette[color_idx].clone()
        color_rgb += torch.randn(3, device=device) * 0.03
        color_rgb = color_rgb.clamp(0, 1)
        color = torch.cat([color_rgb, torch.tensor([1.0], device=device)])
        
        # 采样位置（从显著性图中采样）
        positions = sample_from_attention_map(saliency, 1, temperature=0.8)
        start_x, start_y = positions[0][0].item(), positions[0][1].item()
        
        # 笔画方向（随机，底色阶段不需要精确方向）
        angle = np.random.uniform(0, 2 * np.pi)
        length = np.random.uniform(0.1, 0.4)
        
        # 构建控制点
        raw_points = torch.zeros(num_control_points, 2, device=device)
        for j in range(num_control_points):
            t = j / (num_control_points - 1)
            offset_x = start_x + length * t * np.cos(angle) + np.random.randn() * 0.02
            offset_y = start_y + length * t * np.sin(angle) + np.random.randn() * 0.02
            raw_points[j] = torch.tensor([offset_x, offset_y])
        
        raw_points = torch.logit(torch.clamp(raw_points, 0.01, 0.99))
        
        width = np.random.uniform(min_width, max_width)
        
        stroke = BrushStroke(
            num_control_points=num_control_points,
            canvas_size=canvas_size,
            init_width=width,
            init_color=color,
            init_opacity=0.8,
            device=device,
        )
        stroke.raw_control_points = nn.Parameter(raw_points)
        
        strokes.append(stroke)
        # 底色阶段使用马克笔和水彩
        brush_names.append(["marker", "watercolor", "airbrush"][i % 3])
    
    return strokes, brush_names


def initialize_strokes_stage2(
    target_image: torch.Tensor,
    canvas_size: Tuple[int, int] = (640, 480),
    num_strokes: int = 100,
    max_width: float = 15.0,
    min_width: float = 5.0,
    num_control_points: int = 7,
    device: str = "cpu",
    rendered_image: Optional[torch.Tensor] = None,
) -> Tuple[List[BrushStroke], List[str]]:
    """
    Stage 2: 形体刻画初始化
    
    基于边缘检测和残差注意力：
    - 在边缘区域放置笔画（模拟画家勾勒轮廓）
    - 在残差大的区域放置笔画（补充底色未覆盖的区域）
    - 笔画方向沿边缘方向
    """
    # 计算边缘图
    edges = extract_edges_torch(target_image)
    
    # 计算残差注意力
    if rendered_image is not None:
        # Resize rendered_image to match target_image
        if rendered_image.shape != target_image.shape:
            if rendered_image.dim() == 3:
                rendered_image = rendered_image.unsqueeze(0)
        _target = target_image.unsqueeze(0) if target_image.dim() == 3 else target_image
        rendered_image = F.interpolate(rendered_image, size=_target.shape[2:], mode='bilinear', align_corners=False)
        if target_image.dim() == 3:
            rendered_image = rendered_image.squeeze(0)
        residual = (target_image - rendered_image).abs().mean(dim=0)
        if residual.max() > 0:
            residual = residual / residual.max()
        # 组合边缘和残差
        attention = edges * 0.4 + residual * 0.6
    else:
        attention = edges
    strokes = []
    brush_names = []
    
    for i in range(num_strokes):
        # 从注意力图采样位置
        positions = sample_from_attention_map(attention, 1, temperature=0.5)
        start_x, start_y = positions[0][0].item(), positions[0][1].item()
        
        # 沿边缘方向放置笔画
        stroke_angle = compute_local_gradient_direction(target_image, (start_x, start_y))
        
        # 中等长度
        length = np.random.uniform(0.05, 0.2)
        
        # 采样颜色
        img_y = int(start_y * (target_image.shape[1] - 1))
        img_x = int(start_x * (target_image.shape[2] - 1))
        color_rgb = target_image[:, img_y, img_x].clone()
        color_rgb += torch.randn(3, device=device) * 0.02
        color_rgb = color_rgb.clamp(0, 1)
        color = torch.cat([color_rgb, torch.tensor([1.0], device=device)])
        
        # 构建控制点
        raw_points = torch.zeros(num_control_points, 2, device=device)
        for j in range(num_control_points):
            t = j / (num_control_points - 1)
            offset_x = start_x + length * t * np.cos(stroke_angle) + np.random.randn() * 0.01
            offset_y = start_y + length * t * np.sin(stroke_angle) + np.random.randn() * 0.01
            raw_points[j] = torch.tensor([offset_x, offset_y])
        
        raw_points = torch.logit(torch.clamp(raw_points, 0.01, 0.99))
        
        width = np.random.uniform(min_width, max_width)
        
        stroke = BrushStroke(
            num_control_points=num_control_points,
            canvas_size=canvas_size,
            init_width=width,
            init_color=color,
            init_opacity=0.85,
            device=device,
        )
        stroke.raw_control_points = nn.Parameter(raw_points)
        
        strokes.append(stroke)
        # 刻画阶段使用压感笔和马克笔
        brush_names.append(["pressure", "pressure_sharp", "marker"][i % 3])
    
    return strokes, brush_names


def initialize_strokes_stage3(
    target_image: torch.Tensor,
    canvas_size: Tuple[int, int] = (640, 480),
    num_strokes: int = 300,
    max_width: float = 3.0,
    min_width: float = 1.0,
    num_control_points: int = 9,
    device: str = "cpu",
    rendered_image: Optional[torch.Tensor] = None,
) -> Tuple[List[BrushStroke], List[str]]:
    """
    Stage 3: 细节线稿初始化
    
    基于高频信息和残差注意力：
    - 在高频区域放置细笔画（模拟画家添加纹理和细节）
    - 在残差大的区域放置笔画（补充前面阶段未覆盖的细节）
    - 笔画方向沿梯度方向
    """
    # 计算高频信息
    high_freq = extract_high_frequency(target_image)
    
    # 计算残差注意力
    if rendered_image is not None:
        # Resize rendered_image to match target_image
        if rendered_image.shape != target_image.shape:
            if rendered_image.dim() == 3:
                rendered_image = rendered_image.unsqueeze(0)
        _target = target_image.unsqueeze(0) if target_image.dim() == 3 else target_image
        rendered_image = F.interpolate(rendered_image, size=_target.shape[2:], mode='bilinear', align_corners=False)
        if target_image.dim() == 3:
            rendered_image = rendered_image.squeeze(0)
        residual = (target_image - rendered_image).abs().mean(dim=0)
        if residual.max() > 0:
            residual = residual / residual.max()
        attention = high_freq * 0.3 + residual * 0.7
    else:
        attention = high_freq
    strokes = []
    brush_names = []
    
    for i in range(num_strokes):
        # 从注意力图采样位置
        positions = sample_from_attention_map(attention, 1, temperature=0.3)
        start_x, start_y = positions[0][0].item(), positions[0][1].item()
        
        # 沿梯度方向放置笔画
        stroke_angle = compute_local_gradient_direction(target_image, (start_x, start_y))
        
        # 短笔画
        length = np.random.uniform(0.02, 0.1)
        
        # 采样颜色
        img_y = int(start_y * (target_image.shape[1] - 1))
        img_x = int(start_x * (target_image.shape[2] - 1))
        color_rgb = target_image[:, img_y, img_x].clone()
        color_rgb += torch.randn(3, device=device) * 0.02
        color_rgb = color_rgb.clamp(0, 1)
        color = torch.cat([color_rgb, torch.tensor([1.0], device=device)])
        
        # 构建控制点
        raw_points = torch.zeros(num_control_points, 2, device=device)
        for j in range(num_control_points):
            t = j / (num_control_points - 1)
            offset_x = start_x + length * t * np.cos(stroke_angle) + np.random.randn() * 0.005
            offset_y = start_y + length * t * np.sin(stroke_angle) + np.random.randn() * 0.005
            raw_points[j] = torch.tensor([offset_x, offset_y])
        
        raw_points = torch.logit(torch.clamp(raw_points, 0.01, 0.99))
        
        width = np.random.uniform(min_width, max_width)
        
        stroke = BrushStroke(
            num_control_points=num_control_points,
            canvas_size=canvas_size,
            init_width=width,
            init_color=color,
            init_opacity=0.9,
            device=device,
        )
        stroke.raw_control_points = nn.Parameter(raw_points)
        
        strokes.append(stroke)
        # 细节阶段使用铅笔和细压感笔
        brush_names.append("pencil" if i % 3 != 0 else "pressure_sharp")
    
    return strokes, brush_names
