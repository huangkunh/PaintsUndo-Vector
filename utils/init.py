"""
笔画初始化策略 - 基于图像分析的智能初始化

根据目标图像的特征，智能初始化笔画参数。
不同阶段使用不同的初始化策略：
- 底色阶段：基于颜色聚类的大面积覆盖
- 刻画阶段：基于边缘检测和显著性区域的笔画放置
- 细节阶段：基于高频信息和高梯度区域的精细笔画

关键改进：
1. 颜色聚类初始化：从目标图像提取主色调，按颜色区域放置笔画
2. 边缘引导初始化：沿边缘方向放置笔画，模拟画家的轮廓勾勒
3. 显著性引导：在视觉显著区域放置更多笔画
4. 渐进式细化：每个阶段在前一阶段的残差上初始化
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from brushes.base import BrushStroke


def image_to_tensor(image_path: str, size: Tuple[int, int], device: str = "cpu") -> torch.Tensor:
    """加载图像并转换为张量 [C, H, W]，值域 [0, 1]"""
    from PIL import Image
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
    从图像中提取主要颜色（K-Means 聚类）。
    
    模拟画家调色板：提取图像的主要色调，
    用于初始化底色阶段的笔画颜色。
    """
    import cv2
    
    img_np = image.permute(1, 2, 0).cpu().numpy()
    pixels = img_np.reshape(-1, 3).astype(np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels, num_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    
    centers = torch.from_numpy(centers).to(image.device)
    
    # 按颜色区域大小排序（大区域优先）
    label_counts = np.bincount(labels.flatten())
    sorted_indices = np.argsort(-label_counts)
    centers = centers[sorted_indices]
    
    return centers


def extract_edges(
    image: torch.Tensor,
    low_threshold: int = 50,
    high_threshold: int = 150,
) -> torch.Tensor:
    """
    Canny 边缘检测。
    
    用于刻画阶段：沿边缘方向放置笔画，
    模拟画家的轮廓勾勒过程。
    """
    import cv2
    
    img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    return torch.from_numpy(edges.astype(np.float32) / 255.0).to(image.device)


def extract_saliency(
    image: torch.Tensor,
) -> torch.Tensor:
    """
    简单的显著性检测（基于频谱残差）。
    
    在视觉显著区域放置更多笔画，
    模拟画家对重要区域的精细刻画。
    """
    import cv2
    
    img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # 频谱残差显著性检测
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(gray)
    
    if not success:
        # 回退到简单的中心偏好
        H, W = gray.shape
        y, x = np.mgrid[0:H, 0:W]
        cx, cy = W / 2, H / 2
        saliency_map = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * (min(W, H) / 3) ** 2))
    
    return torch.from_numpy(saliency_map.astype(np.float32)).to(image.device)


def extract_high_frequency(
    image: torch.Tensor,
    blur_kernel: int = 15,
) -> torch.Tensor:
    """
    提取高频信息（细节层）。
    
    原图 - 低通滤波 = 高频细节
    用于细节阶段：在高频信息丰富的区域放置精细笔画。
    """
    import cv2
    
    img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(img_np, (blur_kernel, blur_kernel), 0)
    high_freq = img_np.astype(np.float32) - blurred.astype(np.float32)
    high_freq = np.abs(high_freq).mean(axis=2)
    high_freq = high_freq / (high_freq.max() + 1e-8)
    
    return torch.from_numpy(high_freq).to(image.device)


def compute_gradient_map(image: torch.Tensor) -> torch.Tensor:
    """
    计算图像梯度图（Sobel 算子）。
    
    用于确定笔画方向：笔画应沿梯度垂直方向（即沿边缘方向）放置，
    模拟画家沿轮廓线绘画的习惯。
    """
    gray = image.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, H, W]
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=torch.float32, device=image.device).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=torch.float32, device=image.device).reshape(1, 1, 3, 3)
    
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    
    gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
    gradient_direction = torch.atan2(grad_y, grad_x)
    
    return gradient_magnitude.squeeze(), gradient_direction.squeeze()


def sample_point_from_map(prob_map: torch.Tensor) -> Tuple[float, float]:
    """
    从概率图中采样一个点。
    
    概率值越高的区域被采样的概率越大，
    模拟画家在重要区域放置更多笔画。
    """
    prob = prob_map.cpu().numpy().flatten()
    prob = prob / (prob.sum() + 1e-8)
    idx = np.random.choice(len(prob), p=prob)
    H, W = prob_map.shape
    y = idx // W
    x = idx % W
    return x / W, y / H  # 归一化坐标


def initialize_strokes_stage1(
    target_image: torch.Tensor,
    num_strokes: int = 20,
    canvas_size: Tuple[int, int] = (640, 480),
    num_control_points: int = 5,
    max_width: float = 50.0,
    min_width: float = 20.0,
    device: str = "cpu",
) -> Tuple[List[BrushStroke], List[str]]:
    """
    Stage 1: 铺底色初始化
    
    策略：
    1. 提取图像主色调（K-Means 聚类）
    2. 按颜色区域大小分配笔画数量
    3. 在每个颜色区域内放置粗大的笔画
    4. 使用马克笔和水彩笔刷（大面积覆盖）
    
    模拟人类画家：先用大号画笔铺满底色
    """
    # 提取主色调
    palette = extract_color_palette(target_image, num_colors=min(8, num_strokes))
    
    # 计算颜色区域图
    img_flat = target_image.permute(1, 2, 0).reshape(-1, 3)
    palette_flat = palette
    
    # 为每个像素分配最近的调色板颜色
    dists = torch.cdist(img_flat.unsqueeze(0), palette_flat.unsqueeze(0)).squeeze(0)
    labels = dists.argmin(dim=1)
    
    H, W = target_image.shape[1], target_image.shape[2]
    label_map = labels.reshape(H, W)
    
    # 按区域大小分配笔画数量
    label_counts = torch.bincount(labels, minlength=len(palette))
    total_count = label_counts.sum().float()
    stroke_counts = (label_counts.float() / total_count * num_strokes).long()
    stroke_counts = stroke_counts.clamp(min=1)
    
    # 调整总数
    while stroke_counts.sum() > num_strokes:
        max_idx = stroke_counts.argmax()
        stroke_counts[max_idx] -= 1
    while stroke_counts.sum() < num_strokes:
        min_idx = (label_counts.float() / (stroke_counts.float() + 1)).argmax()
        stroke_counts[min_idx] += 1
    
    strokes = []
    brush_names = []
    
    for color_idx in range(len(palette)):
        count = stroke_counts[color_idx].item()
        if count == 0:
            continue
        
        color_rgb = palette[color_idx]
        color = torch.cat([color_rgb, torch.tensor([0.85], device=device)])  # 略微透明
        
        # 获取该颜色区域的像素位置
        mask = (label_map == color_idx)
        positions = mask.nonzero().float()  # [N, 2] (y, x)
        
        if positions.shape[0] < 2:
            continue
        
        for _ in range(count):
            # 在颜色区域内随机采样起点
            idx = np.random.randint(0, positions.shape[0])
            start_y = positions[idx, 0].item() / H
            start_x = positions[idx, 1].item() / W
            
            # 随机方向和长度
            angle = np.random.uniform(0, 2 * np.pi)
            length = np.random.uniform(0.1, 0.4)
            
            # 构建控制点
            raw_points = torch.zeros(num_control_points, 2, device=device)
            for j in range(num_control_points):
                t = j / (num_control_points - 1)
                offset_x = start_x + length * t * np.cos(angle) + np.random.randn() * 0.02
                offset_y = start_y + length * t * np.sin(angle) + np.random.randn() * 0.02
                raw_points[j] = torch.tensor([offset_x, offset_y])
            
            # sigmoid 逆变换
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
            # 底色阶段交替使用马克笔和水彩
            brush_names.append("marker" if _ % 2 == 0 else "watercolor")
    
    return strokes, brush_names


def initialize_strokes_stage2(
    target_image: torch.Tensor,
    residual_image: Optional[torch.Tensor] = None,
    num_strokes: int = 100,
    canvas_size: Tuple[int, int] = (640, 480),
    num_control_points: int = 7,
    max_width: float = 15.0,
    min_width: float = 5.0,
    device: str = "cpu",
) -> Tuple[List[BrushStroke], List[str]]:
    """
    Stage 2: 形体刻画初始化
    
    策略：
    1. 使用边缘检测确定轮廓位置
    2. 使用梯度方向确定笔画方向（沿边缘方向）
    3. 使用显著性图确定笔画密度
    4. 使用压感笔刷（变宽效果更自然）
    
    模拟人类画家：在底色基础上勾勒形体轮廓和色块过渡
    """
    # 使用残差图像（如果有）或原始图像
    ref_image = residual_image if residual_image is not None else target_image
    
    # 提取边缘和梯度
    edges = extract_edges(ref_image)
    grad_mag, grad_dir = compute_gradient_map(ref_image)
    
    # 计算显著性
    try:
        saliency = extract_saliency(ref_image)
    except Exception:
        # 回退到梯度幅度作为显著性
        saliency = grad_mag / (grad_mag.max() + 1e-8)
    
    # 综合概率图：边缘 + 梯度 + 显著性
    prob_map = edges * 0.4 + (grad_mag / (grad_mag.max() + 1e-8)) * 0.3 + saliency * 0.3
    prob_map = prob_map / (prob_map.max() + 1e-8)
    
    strokes = []
    brush_names = []
    
    for i in range(num_strokes):
        # 从概率图采样起点
        start_x, start_y = sample_point_from_map(prob_map)
        
        # 获取该点的梯度方向（笔画应沿边缘方向）
        py = int(start_y * (edges.shape[0] - 1))
        px = int(start_x * (edges.shape[1] - 1))
        local_grad_dir = grad_dir[py, px].item()
        
        # 笔画方向 = 梯度垂直方向（沿边缘）
        stroke_angle = local_grad_dir + np.pi / 2 + np.random.randn() * 0.3
        
        # 笔画长度
        length = np.random.uniform(0.05, 0.2)
        
        # 采样颜色
        img_y = int(start_y * (target_image.shape[1] - 1))
        img_x = int(start_x * (target_image.shape[2] - 1))
        color_rgb = target_image[:, img_y, img_x].clone()
        color_rgb += torch.randn(3, device=device) * 0.03
        color_rgb = color_rgb.clamp(0, 1)
        color = torch.cat([color_rgb, torch.tensor([0.9], device=device)])
        
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
        brush_names.append("pressure" if i % 3 != 0 else "marker")
    
    return strokes, brush_names


def initialize_strokes_stage3(
    target_image: torch.Tensor,
    residual_image: Optional[torch.Tensor] = None,
    num_strokes: int = 300,
    canvas_size: Tuple[int, int] = (640, 480),
    num_control_points: int = 9,
    max_width: float = 3.0,
    min_width: float = 1.0,
    device: str = "cpu",
) -> Tuple[List[BrushStroke], List[str]]:
    """
    Stage 3: 细节线稿初始化
    
    策略：
    1. 使用高频信息图确定细节区域
    2. 使用精细的梯度方向引导
    3. 使用铅笔和细压感笔刷
    4. 更短、更细的笔画
    
    模拟人类画家：在形体基础上添加细节线条和纹理
    """
    ref_image = residual_image if residual_image is not None else target_image
    
    # 提取高频信息
    high_freq = extract_high_frequency(ref_image)
    
    # 提取边缘
    edges = extract_edges(ref_image, low_threshold=30, high_threshold=100)
    
    # 梯度方向
    grad_mag, grad_dir = compute_gradient_map(ref_image)
    
    # 综合概率图
    prob_map = high_freq * 0.5 + edges * 0.3 + (grad_mag / (grad_mag.max() + 1e-8)) * 0.2
    prob_map = prob_map / (prob_map.max() + 1e-8)
    
    strokes = []
    brush_names = []
    
    for i in range(num_strokes):
        # 从概率图采样
        start_x, start_y = sample_point_from_map(prob_map)
        
        # 获取梯度方向
        py = int(start_y * (edges.shape[0] - 1))
        px = int(start_x * (edges.shape[1] - 1))
        local_grad_dir = grad_dir[py, px].item()
        
        # 笔画方向
        stroke_angle = local_grad_dir + np.pi / 2 + np.random.randn() * 0.2
        
        # 更短的笔画
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
