"""
行为损失模块 - 让绘画过程更像人类

核心思想：
- 人类画师在每一步的中间结果都有特定的视觉特征
- 行为损失约束：当前渲染结果不仅要像目标图，还要像人类在该步骤的中间结果
- 中间帧通过 PaintsUndo 风格的关键帧生成器模拟

损失设计：
- L_result = ||render(strokes) - target||  （结果损失：最终要像目标图）
- L_behavior = ||render(strokes_at_step_t) - human_frame_at_step_t||  （行为损失：中间过程要像人类）
- L_total = L_result + λ * L_behavior
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict

from core.semantic.layers import LayerType, HUMAN_LAYER_ORDER, LAYER_ALPHA_MAP


class HumanFrameGenerator:
    """
    人类中间帧生成器
    
    模拟 PaintsUndo 的单帧模型：
    给定目标图像和操作步骤 t，生成该步骤时人类画师的"中间帧"。
    
    实现方式（不依赖深度学习模型）：
    - 用多尺度模糊模拟不同阶段的"完成度"
    - 早期阶段：极度模糊，只有大色块
    - 中期阶段：中等模糊，可辨认形体
    - 后期阶段：轻微模糊，接近完成
    """
    
    def __init__(self, target: torch.Tensor, num_steps: int = 8):
        """
        Args:
            target: 目标图像 [3, H, W]
            num_steps: 总步骤数（对应 PaintsUndo 的 operation_step 0-999）
        """
        self.target = target
        self.num_steps = num_steps
        self._cache = {}
    
    def get_frame(self, step: int) -> torch.Tensor:
        """
        获取指定步骤的人类中间帧
        
        Args:
            step: 步骤 0 到 num_steps-1
        Returns:
            [3, H, W] 中间帧
        """
        if step in self._cache:
            return self._cache[step]
        
        if step >= self.num_steps - 1:
            return self.target
        
        # 完成度：0.0（空白）到 1.0（完成）
        progress = step / max(self.num_steps - 1, 1)
        
        # 根据完成度生成中间帧
        frame = self._generate_frame_at_progress(progress)
        self._cache[step] = frame
        return frame
    
    def _generate_frame_at_progress(self, progress: float) -> torch.Tensor:
        """
        根据完成度生成中间帧
        
        模拟人类画师的绘画过程：
        - progress < 0.1: 几乎空白，只有极模糊的色块
        - progress 0.1-0.3: 粗略色块，可辨认大色域
        - progress 0.3-0.5: 形体出现，但细节模糊
        - progress 0.5-0.7: 细节开始出现
        - progress 0.7-0.9: 接近完成，只有小调整
        - progress > 0.9: 几乎完成
        """
        C, H, W = self.target.shape
        
        if progress < 0.05:
            # 极早期：只有背景色
            avg_color = self.target.mean(dim=(1, 2))
            frame = avg_color.unsqueeze(1).unsqueeze(2).expand(C, H, W)
            return frame
        
        # 使用多尺度模糊模拟不同完成度
        # progress 越小，模糊越强
        sigma = max(1.0, (1.0 - progress) * 30.0)
        kernel_size = int(sigma * 4) | 1  # 确保奇数
        
        # 高斯模糊
        blurred = self._gaussian_blur(self.target, kernel_size, sigma)
        
        # 混合模糊和清晰图像
        # progress 越大，清晰图像占比越高
        frame = blurred * (1.0 - progress ** 0.5) + self.target * (progress ** 0.5)
        
        # 早期阶段降低饱和度（模拟"先铺灰色底，再上色"）
        if progress < 0.3:
            gray = 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]
            gray = gray.unsqueeze(0).expand(3, H, W)
            desat = 1.0 - progress / 0.3  # 去饱和度
            frame = frame * (1 - desat * 0.5) + gray * (desat * 0.5)
        
        return frame.clamp(0, 1)
    
    def _gaussian_blur(self, img: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
        """高斯模糊"""
        C, H, W = img.shape
        x = torch.arange(kernel_size, dtype=torch.float32, device=img.device) - kernel_size // 2
        gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel_1d = gauss / gauss.sum()
        
        # 分离卷积
        img_4d = img.unsqueeze(0)  # [1, C, H, W]
        
        # 水平模糊
        kh = kernel_1d.view(1, 1, 1, kernel_size).expand(C, 1, 1, kernel_size)
        pad_h = kernel_size // 2
        blurred = F.conv2d(img_4d, kh, padding=(0, pad_h), groups=C)
        
        # 垂直模糊
        kv = kernel_1d.view(1, 1, kernel_size, 1).expand(C, 1, kernel_size, 1)
        pad_v = kernel_size // 2
        blurred = F.conv2d(blurred, kv, padding=(pad_v, 0), groups=C)
        
        return blurred.squeeze(0).clamp(0, 1)


class BehaviorLoss(nn.Module):
    """
    行为损失 - 约束绘画过程符合人类行为
    
    L_behavior = Σ_t λ_t * ||render(strokes[:t]) - human_frame_t||_perceptual
    
    其中：
    - t 是绘画步骤
    - λ_t 是步骤权重（早期步骤权重更高，因为人类画师的早期步骤更可预测）
    - human_frame_t 是人类在步骤 t 时的中间帧
    """
    
    def __init__(self, target: torch.Tensor, num_steps: int = 8, device: str = "cpu"):
        super().__init__()
        self.frame_generator = HumanFrameGenerator(target, num_steps)
        self.num_steps = num_steps
        self.device = device
        
        # 步骤权重：早期步骤权重更高
        weights = torch.tensor([max(0.5, 1.0 - i / num_steps) for i in range(num_steps)])
        self.register_buffer('step_weights', weights)
    
    def forward(self, rendered: torch.Tensor, step: int, 
                target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算行为损失
        
        Args:
            rendered: 当前渲染结果 [3, H, W]
            step: 当前步骤
            target: 目标图像（可选，用于结果损失）
        Returns:
            行为损失值
        """
        if step >= self.num_steps:
            step = self.num_steps - 1
        
        # 获取人类中间帧
        human_frame = self.frame_generator.get_frame(step)
        
        # 感知损失（简化版：L1 + 结构相似性）
        l1_loss = F.l1_loss(rendered, human_frame)
        
        # 结构损失
        if rendered.dim() == 3:
            r = rendered.unsqueeze(0)
            h = human_frame.unsqueeze(0)
        else:
            r = rendered
            h = human_frame
        
        # 简化 SSIM 差异
        ws = 7
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        mu_r = F.avg_pool2d(r, ws, stride=1, padding=ws // 2)
        mu_h = F.avg_pool2d(h, ws, stride=1, padding=ws // 2)
        sigma_r = F.avg_pool2d(r ** 2, ws, stride=1, padding=ws // 2) - mu_r ** 2
        sigma_h = F.avg_pool2d(h ** 2, ws, stride=1, padding=ws // 2) - mu_h ** 2
        sigma_rh = F.avg_pool2d(r * h, ws, stride=1, padding=ws // 2) - mu_r * mu_h
        ssim = ((2 * mu_r * mu_h + C1) * (2 * sigma_rh + C2)) / \
               ((mu_r ** 2 + mu_h ** 2 + C1) * (sigma_r + sigma_h + C2))
        ssim_loss = 1.0 - ssim.mean()
        
        # 加权组合
        weight = self.step_weights[step]
        behavior_loss = weight * (l1_loss + 0.5 * ssim_loss)
        
        return behavior_loss


class MultiStageLoss(nn.Module):
    """
    多阶段组合损失
    
    L_total = L_result + λ_behavior * L_behavior + λ_rhythm * L_rhythm
    
    其中：
    - L_result: 结果损失（像素+感知+SSIM）
    - L_behavior: 行为损失（中间帧匹配）
    - L_rhythm: 节奏损失（笔画长度/密度变化）
    """
    
    def __init__(self, target: torch.Tensor, device: str = "cpu", lightweight: bool = False,
                 lambda_behavior: float = 0.3, lambda_rhythm: float = 0.01):
        super().__init__()
        self.device = device
        self.lightweight = lightweight
        self.lambda_behavior = lambda_behavior
        self.lambda_rhythm = lambda_rhythm
        
        # 结果损失
        from core.losses import CombinedLoss
        self.result_loss = CombinedLoss(device=device, lightweight=lightweight)
        self.result_loss.cache_target(target)
        
        # 行为损失
        self.behavior_loss = BehaviorLoss(target, num_steps=8, device=device)
    
    def forward(self, rendered: torch.Tensor, target: torch.Tensor,
                strokes: List, step: int = 0, stage: int = 0) -> Tuple[torch.Tensor, Dict]:
        """
        计算总损失
        
        Args:
            rendered: 渲染结果
            target: 目标图像
            strokes: 笔画列表
            step: 当前绘画步骤
            stage: 当前优化阶段
        Returns:
            (总损失, 损失字典)
        """
        # 结果损失
        result_loss, result_dict = self.result_loss(rendered, target, strokes, stage=stage)
        
        # 行为损失
        behavior_loss = self.behavior_loss(rendered, step, target)
        
        # 节奏损失（笔画长度变化应该有节奏感）
        rhythm_loss = torch.tensor(0.0, device=self.device)
        if strokes and len(strokes) > 2:
            lengths = []
            for s in strokes:
                if hasattr(s, 'get_length'):
                    lengths.append(s.get_length())
            if len(lengths) > 2:
                lengths_t = torch.stack(lengths)
                # 长度变化率不应该太剧烈
                diffs = lengths_t[1:] - lengths_t[:-1]
                rhythm_loss = diffs.abs().mean() * self.lambda_rhythm
        
        # 总损失
        total = result_loss + self.lambda_behavior * behavior_loss + rhythm_loss
        
        loss_dict = {
            **result_dict,
            "behavior": behavior_loss.item(),
            "rhythm": rhythm_loss.item(),
            "total": total.item(),
        }
        
        return total, loss_dict
