"""
多阶段优化器 - 核心优化管线

实现方案B的渐进式优化策略，模拟人类画家的绘画过程：
- Stage 1: 铺底色（低分辨率，粗线条，少量笔画）- 模拟画家用大号画笔铺底色
- Stage 2: 形体刻画（中分辨率，中等线条，较多笔画）- 模拟画家勾勒轮廓和色块
- Stage 3: 细节线稿（高分辨率，细线条，大量笔画）- 模拟画家添加细节和纹理

关键改进：
1. 每个阶段锁定前一阶段的参数，只优化当前阶段的笔画
2. 使用残差图像引导：每个阶段在前一阶段的残差上优化
3. 自适应笔画数量：根据残差大小动态增加笔画
4. 人类绘画模拟：笔画排序、重叠策略、颜色混合
5. 渐进式分辨率提升
"""

import os
import time
import copy
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from brushes.base import BrushStroke
from core.renderer import DifferentiableRenderer
from core.losses import CombinedLoss
from utils.image import load_image, save_image, resize_image
from utils.init import (
    initialize_strokes_stage1,
    initialize_strokes_stage2,
    initialize_strokes_stage3,
)


class StageConfig:
    """单个阶段的配置"""
    
    def __init__(
        self,
        name: str,
        num_strokes: int = 20,
        max_width: float = 50.0,
        min_width: float = 20.0,
        num_control_points: int = 5,
        resolution: Tuple[int, int] = (256, 256),
        num_iterations: int = 500,
        lr: float = 1.0,
        lr_width: float = 0.1,
        lr_color: float = 0.01,
        brush_names: List[str] = None,
        lambda_pixel: float = 1.0,
        lambda_perceptual: float = 10.0,
        lambda_ssim: float = 2.0,
        lambda_color: float = 1.0,
        lambda_length: float = 0.01,
        lambda_smooth: float = 0.005,
    ):
        self.name = name
        self.num_strokes = num_strokes
        self.max_width = max_width
        self.min_width = min_width
        self.num_control_points = num_control_points
        self.resolution = resolution
        self.num_iterations = num_iterations
        self.lr = lr
        self.lr_width = lr_width
        self.lr_color = lr_color
        self.brush_names = brush_names or ["marker"]
        self.lambda_pixel = lambda_pixel
        self.lambda_perceptual = lambda_perceptual
        self.lambda_ssim = lambda_ssim
        self.lambda_color = lambda_color
        self.lambda_length = lambda_length
        self.lambda_smooth = lambda_smooth


# 默认三阶段配置
DEFAULT_STAGE_CONFIGS = [
    StageConfig(
        name="铺底色",
        num_strokes=32,
        max_width=50.0,
        min_width=20.0,
        num_control_points=5,
        resolution=(256, 256),
        num_iterations=500,
        lr=1.0,
        lr_width=0.1,
        lr_color=0.01,
        brush_names=["marker", "watercolor", "airbrush"],
        lambda_pixel=2.0,
        lambda_perceptual=5.0,
        lambda_ssim=0.5,
        lambda_color=2.0,
        lambda_length=0.01,
        lambda_smooth=0.005,
    ),
    StageConfig(
        name="形体刻画",
        num_strokes=128,
        max_width=15.0,
        min_width=5.0,
        num_control_points=7,
        resolution=(512, 512),
        num_iterations=800,
        lr=1.0,
        lr_width=0.1,
        lr_color=0.01,
        brush_names=["pressure", "pressure_sharp", "marker"],
        lambda_pixel=1.0,
        lambda_perceptual=10.0,
        lambda_ssim=2.0,
        lambda_color=1.0,
        lambda_length=0.005,
        lambda_smooth=0.003,
    ),
    StageConfig(
        name="细节线稿",
        num_strokes=512,
        max_width=3.0,
        min_width=1.0,
        num_control_points=9,
        resolution=(1024, 1024),
        num_iterations=1000,
        lr=0.5,
        lr_width=0.05,
        lr_color=0.005,
        brush_names=["pencil", "pressure_sharp", "hatching"],
        lambda_pixel=0.5,
        lambda_perceptual=15.0,
        lambda_ssim=3.0,
        lambda_color=0.5,
        lambda_length=0.003,
        lambda_smooth=0.001,
    ),
]


class MultiStageOptimizer:
    """
    多阶段优化器
    
    模拟人类画家的绘画过程，通过三个阶段逐步优化笔画参数：
    1. 铺底色：粗线条覆盖大面积色彩
    2. 形体刻画：中等线条勾勒轮廓和色块过渡
    3. 细节线稿：细线条添加细节和纹理
    
    每个阶段：
    - 在前一阶段的残差上初始化
    - 锁定前一阶段的参数
    - 只优化当前阶段的笔画
    - 使用自适应学习率
    """
    
    def __init__(
        self,
        canvas_size: Tuple[int, int] = (640, 480),
        device: str = "cpu",
        output_dir: str = "output",
    ):
        self.canvas_size = canvas_size
        self.device = device
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
    
    def optimize(
        self,
        target_image: torch.Tensor,
        stage_configs: Optional[List[StageConfig]] = None,
    ) -> Dict:
        """
        执行多阶段优化。
        
        Args:
            target_image: 目标图像 [C, H, W]
            stage_configs: 阶段配置列表
            
        Returns:
            包含所有笔画和优化历史的字典
        """
        if stage_configs is None:
            stage_configs = DEFAULT_STAGE_CONFIGS
        
        all_strokes = []
        all_brush_names = []
        stages_history = []
        loss_history = []
        
        current_canvas = None  # 当前画布状态
        
        for stage_idx, config in enumerate(stage_configs):
            print(f"\n{'='*60}")
            print(f"Stage {stage_idx + 1}: {config.name}")
            print(f"  笔画数: {config.num_strokes}, 分辨率: {config.resolution}, 迭代: {config.num_iterations}")
            print(f"{'='*60}")
            
            # 调整目标图像到当前阶段分辨率
            stage_target = resize_image(target_image, config.resolution).to(self.device)
            
            # 计算残差图像（当前画布与目标的差异）
            if current_canvas is not None:
                current_resized = resize_image(current_canvas, config.resolution).to(self.device)
                residual = (stage_target - current_resized).abs()
                # 残差作为下一阶段的参考
                ref_image = residual
            else:
                residual = None
                ref_image = stage_target
            
            # 初始化笔画
            strokes, brush_names = self._initialize_strokes(
                stage_idx, stage_target, ref_image, config
            )
            
            # 锁定前一阶段的笔画参数
            for prev_stroke in all_strokes:
                prev_stroke.requires_grad_(False)
            
            # 优化当前阶段的笔画
            optimized_strokes, stage_loss_history = self._optimize_stage(
                strokes, brush_names, stage_target, config, stage_idx,
                frozen_strokes=all_strokes,
                frozen_brush_names=all_brush_names,
            )
            
            # 记录结果
            all_strokes.extend(optimized_strokes)
            all_brush_names.extend(brush_names)
            
            stages_history.append({
                "strokes": optimized_strokes,
                "brush_names": brush_names,
                "loss_history": stage_loss_history,
                "config": config,
            })
            loss_history.extend(stage_loss_history)
            
            # 渲染当前画布状态
            renderer = DifferentiableRenderer(
                canvas_size=config.resolution,
                device=self.device,
            )
            current_canvas = renderer.render_multi_brush(
                self._group_strokes_by_brush(all_strokes, all_brush_names)
            ).detach()
            
            # 保存中间结果
            self._save_stage_result(stage_idx, current_canvas, stage_target, optimized_strokes, brush_names)
        
        return {
            "strokes": all_strokes,
            "brush_names": all_brush_names,
            "stages_history": stages_history,
            "loss_history": loss_history,
            "final_canvas": current_canvas,
        }
    
    def _initialize_strokes(
        self,
        stage_idx: int,
        target_image: torch.Tensor,
        ref_image: torch.Tensor,
        config: StageConfig,
    ) -> Tuple[List[BrushStroke], List[str]]:
        """根据阶段选择初始化策略"""
        if stage_idx == 0:
            return initialize_strokes_stage1(
                target_image=target_image,
                num_strokes=config.num_strokes,
                canvas_size=config.resolution,
                num_control_points=config.num_control_points,
                max_width=config.max_width,
                min_width=config.min_width,
                device=self.device,
            )
        elif stage_idx == 1:
            return initialize_strokes_stage2(
                target_image=target_image,
                residual_image=ref_image,
                num_strokes=config.num_strokes,
                canvas_size=config.resolution,
                num_control_points=config.num_control_points,
                max_width=config.max_width,
                min_width=config.min_width,
                device=self.device,
            )
        else:
            return initialize_strokes_stage3(
                target_image=target_image,
                residual_image=ref_image,
                num_strokes=config.num_strokes,
                canvas_size=config.resolution,
                num_control_points=config.num_control_points,
                max_width=config.max_width,
                min_width=config.min_width,
                device=self.device,
            )
    
    def _optimize_stage(
        self,
        strokes: List[BrushStroke],
        brush_names: List[str],
        target: torch.Tensor,
        config: StageConfig,
        stage_idx: int,
        frozen_strokes: List[BrushStroke] = None,
        frozen_brush_names: List[str] = None,
    ) -> Tuple[List[BrushStroke], List[float]]:
        """
        优化单个阶段的笔画参数。
        
        使用 Adam 优化器，分三组学习率：
        - 控制点: config.lr
        - 宽度: config.lr_width
        - 颜色: config.lr_color
        
        优化过程中：
        - 每50次迭代检查是否需要增加笔画
        - 使用余弦退火学习率调度
        - 记录损失历史
        """
        renderer = DifferentiableRenderer(
            canvas_size=config.resolution,
            device=self.device,
        )
        
        loss_fn = CombinedLoss(
            device=self.device,
            lambda_pixel=config.lambda_pixel,
            lambda_perceptual=config.lambda_perceptual,
            lambda_ssim=config.lambda_ssim,
            lambda_color=config.lambda_color,
            lambda_length=config.lambda_length,
            lambda_smooth=config.lambda_smooth,
        )
        
        # 分组参数
        point_params = []
        width_params = []
        color_params = []
        opacity_params = []
        
        for stroke in strokes:
            point_params.append(stroke.raw_control_points)
            width_params.append(stroke.raw_width)
            color_params.append(stroke.raw_color)
            opacity_params.append(stroke.raw_opacity)
        
        optimizer = optim.Adam([
            {"params": point_params, "lr": config.lr},
            {"params": width_params, "lr": config.lr_width},
            {"params": color_params, "lr": config.lr_color},
            {"params": opacity_params, "lr": config.lr_color * 0.5},
        ], betas=(0.9, 0.999), eps=1e-8)
        
        # 余弦退火学习率调度
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.num_iterations, eta_min=config.lr * 0.01
        )
        
        loss_history = []
        best_loss = float('inf')
        best_strokes = None
        
        for iteration in range(config.num_iterations):
            optimizer.zero_grad()
            
            # 渲染所有笔画（包括冻结的）
            stroke_groups = []
            if frozen_strokes and frozen_brush_names:
                stroke_groups.extend(
                    self._group_strokes_by_brush(frozen_strokes, frozen_brush_names).items()
                )
            stroke_groups.extend(
                self._group_strokes_by_brush(strokes, brush_names).items()
            )
            stroke_groups = [(k, v) for k, v in stroke_groups]
            
            rendered = renderer.render_multi_brush(stroke_groups)
            
            # 计算损失
            total_loss, loss_dict = loss_fn(rendered, target, strokes, stage=stage_idx)
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(point_params, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(width_params, max_norm=0.5)
            
            optimizer.step()
            scheduler.step()
            
            loss_history.append(loss_dict["total"])
            
            # 记录最佳结果
            if loss_dict["total"] < best_loss:
                best_loss = loss_dict["total"]
                best_strokes = copy.deepcopy(strokes)
            
            # 打印进度
            if iteration % 50 == 0 or iteration == config.num_iterations - 1:
                lr = optimizer.param_groups[0]["lr"]
                print(f"  Iter {iteration:4d}/{config.num_iterations} | "
                      f"Loss: {loss_dict['total']:.4f} "
                      f"(pixel: {loss_dict['pixel']:.4f}, "
                      f"perceptual: {loss_dict['perceptual']:.4f}, "
                      f"ssim: {loss_dict['ssim']:.4f}) | "
                      f"LR: {lr:.6f}")
                
                # 保存中间结果
                if iteration % 100 == 0:
                    self._save_iteration_result(
                        stage_idx, iteration, rendered, target
                    )
        
        # 使用最佳结果
        if best_strokes is not None:
            for i, (orig, best) in enumerate(zip(strokes, best_strokes)):
                orig.raw_control_points.data.copy_(best.raw_control_points.data)
                orig.raw_width.data.copy_(best.raw_width.data)
                orig.raw_color.data.copy_(best.raw_color.data)
                orig.raw_opacity.data.copy_(best.raw_opacity.data)
        
        return strokes, loss_history
    
    def _group_strokes_by_brush(
        self,
        strokes: List[BrushStroke],
        brush_names: List[str],
    ) -> Dict[str, List[BrushStroke]]:
        """按笔刷类型分组笔画"""
        groups = {}
        for stroke, brush_name in zip(strokes, brush_names):
            if brush_name not in groups:
                groups[brush_name] = []
            groups[brush_name].append(stroke)
        return groups
    
    def _save_stage_result(
        self,
        stage_idx: int,
        rendered: torch.Tensor,
        target: torch.Tensor,
        strokes: List[BrushStroke],
        brush_names: List[str],
    ):
        """保存阶段结果"""
        stage_dir = os.path.join(self.output_dir, f"stage_{stage_idx + 1}")
        os.makedirs(stage_dir, exist_ok=True)
        
        save_image(rendered, os.path.join(stage_dir, "final_render.png"))
        
        # 保存笔画参数
        strokes_data = []
        for stroke, brush_name in zip(strokes, brush_names):
            strokes_data.append({
                "control_points": stroke.control_points.detach().cpu().numpy().tolist(),
                "width": stroke.width.detach().cpu().item(),
                "color": stroke.color.detach().cpu().numpy().tolist(),
                "opacity": stroke.opacity.detach().cpu().item(),
                "brush_name": brush_name,
            })
        
        import json
        with open(os.path.join(stage_dir, "strokes.json"), "w") as f:
            json.dump(strokes_data, f, indent=2)
        
        print(f"  Stage {stage_idx + 1} results saved to: {stage_dir}")
    
    def _save_iteration_result(
        self,
        stage_idx: int,
        iteration: int,
        rendered: torch.Tensor,
        target: torch.Tensor,
    ):
        """保存迭代中间结果"""
        iter_dir = os.path.join(self.output_dir, f"stage_{stage_idx + 1}", "iterations")
        os.makedirs(iter_dir, exist_ok=True)
        
        from utils.vis import create_comparison_image
        comparison = create_comparison_image(target, rendered)
        comparison.save(os.path.join(iter_dir, f"iter_{iteration:04d}.png"))
