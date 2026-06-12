"""
图像处理工具函数
"""

from typing import Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


def load_image(path: str, size: Tuple[int, int], device: str = "cpu") -> torch.Tensor:
    """加载图像为张量 [C, H, W]，值域 [0, 1]"""
    img = Image.open(path).convert("RGB")
    img = img.resize((size[0], size[1]), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).to(device)


def save_image(tensor: torch.Tensor, path: str):
    """保存张量为图像文件"""
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    arr = (arr * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def resize_image(tensor: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """调整图像大小"""
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    return F.interpolate(tensor, size=(size[1], size[0]), mode='bilinear', align_corners=False).squeeze(0)


def gaussian_blur(tensor: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
    """高斯模糊"""
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    channels = tensor.shape[1]
    sigma = kernel_size / 6.0
    
    # 创建高斯核
    x = torch.arange(kernel_size, device=tensor.device) - kernel_size // 2
    gauss = torch.exp(-x.float() ** 2 / (2 * sigma ** 2))
    kernel_1d = gauss / gauss.sum()
    
    # 分离卷积
    kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
    kernel_2d = kernel_2d.expand(channels, 1, kernel_size, kernel_size)
    
    padding = kernel_size // 2
    result = F.conv2d(tensor, kernel_2d, padding=padding, groups=channels)
    
    return result.squeeze(0)
