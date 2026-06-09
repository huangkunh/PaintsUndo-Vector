"""
多阶段优化器 - 核心优化管线

实现方案B的渐进式优化策略，模拟人类画家的绘画过程：
- Stage 1: 铺底色（低分辨率，粗线条，少量笔画）
- Stage 2: 形体刻画（中分辨率，中等线条，较多笔画）
- Stage 3: 细节线稿（高分辨率，细线条，大量笔画）

关键改进：
1. 每个阶段锁定前一阶段的参数，只优化当前阶段的笔画
2. 使用残差图像引导：每个阶段在前一阶段的残差上优化
3. 自适应笔画增删：根据残差大小动态增加/删除笔画
4. 注意力引导的笔画重初始化：优化停滞时重新初始化低效笔画
5. 课程学习：逐步增加优化难度
6. 人类绘画模拟：笔画排序、重叠策略、颜色混合
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
from core.attention import AttentionMap
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
    
    管理三阶段渐进式优化流程，每个阶段：
    1. 初始化当前阶段的笔画参数
    2. 使用可微渲染和梯度下降优化笔画
    3. 锁定当前阶段的参数，进入下一阶段
    4. 记录绘画历史用于回放
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
        
        # 创建渲染器和损失函数
        self.renderer = DifferentiableRenderer(
            canvas_size=canvas_size,
            device=device,
        )
        self.loss_fn = CombinedLoss(device=device)
        self.attention = AttentionMap(device=device)
    
    def optimize(
        self,
        target_image: torch.Tensor,
        stage_configs: Optional[List[StageConfig]] = None,
        progress_callback=None,
    ) -> Dict:
        """
        执行多阶段优化。
        
        Args:
            target_image: [C, H, W] 目标图像
            stage_configs: 阶段配置列表
            progress_callback: 进度回调函数
            
        Returns:
            包含所有笔画和历史的字典
        """
        if stage_configs is None:
            stage_configs = DEFAULT_STAGE_CONFIGS
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        all_strokes = []
        all_brush_names = []
        stages_history = []
        
        # 保存原始目标图像
        save_image(target_image, os.path.join(self.output_dir, "target.png"))
        
        current_rendered = None
        
        for stage_idx, config in enumerate(stage_configs):
            print(f"\n{'='*60}")
            print(f"Stage {stage_idx + 1}: {config.name}")
            print(f"  笔画数: {config.num_strokes}, 分辨率: {config.resolution}, 迭代: {config.num_iterations}")
            print(f"{'='*60}")
            
            # 调整目标图像分辨率
            target_resized = resize_image(target_image, config.resolution)
            
            # 初始化当前阶段的笔画
            init_fn = [initialize_strokes_stage1, initialize_strokes_stage2, initialize_strokes_stage3][
                min(stage_idx, 2)
            ]
            
            strokes, brush_names = init_fn(
                target_image=target_resized,
                canvas_size=config.resolution,
                num_strokes=config.num_strokes,
                max_width=config.max_width,
                min_width=config.min_width,
                num_control_points=config.num_control_points,
                device=self.device,
                rendered_image=current_rendered,
            )
            
            # 创建当前阶段的渲染器
            stage_renderer = DifferentiableRenderer(
                canvas_size=config.resolution,
                device=self.device,
            )
            
            # 创建当前阶段的损失函数
            stage_loss_fn = CombinedLoss(
                device=self.device,
                lambda_pixel=config.lambda_pixel,
                lambda_perceptual=config.lambda_perceptual,
                lambda_ssim=config.lambda_ssim,
                lambda_color=config.lambda_color,
                lambda_length=config.lambda_length,
                lambda_smooth=config.lambda_smooth,
            )
            
            # 优化当前阶段的笔画
            result = self._optimize_stage(
                stage_idx=stage_idx,
                strokes=strokes,
                brush_names=brush_names,
                target_image=target_resized,
                renderer=stage_renderer,
                loss_fn=stage_loss_fn,
                config=config,
                progress_callback=progress_callback,
            )
            
            # 记录结果
            optimized_strokes = result["strokes"]
            optimized_brush_names = result["brush_names"]
            
            # 渲染当前阶段的最终结果
            stroke_groups = self._group_strokes_by_brush(optimized_strokes, optimized_brush_names)
            current_rendered = stage_renderer.render_multi_brush_with_alpha(stroke_groups)[:3]
            
            # 保存阶段结果
            self._save_stage_result(
                stage_idx, current_rendered, optimized_strokes, optimized_brush_names
            )
            
            # 缩放笔画坐标到原始分辨率
            scaled_strokes = self._scale_strokes_to_canvas(
                optimized_strokes, config.resolution, self.canvas_size
            )
            
            all_strokes.extend(scaled_strokes)
            all_brush_names.extend(optimized_brush_names)
            
            stages_history.append({
                "strokes": optimized_strokes,
                "brush_names": optimized_brush_names,
                "config": config,
            })
            
            # 释放 GPU 内存
            del stage_renderer, stage_loss_fn
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        # 最终渲染
        final_renderer = DifferentiableRenderer(
            canvas_size=self.canvas_size,
            device=self.device,
        )
        stroke_groups = self._group_strokes_by_brush(all_strokes, all_brush_names)
        final_rendered = final_renderer.render_multi_brush(stroke_groups)
        save_image(final_rendered, os.path.join(self.output_dir, "final_render.png"))
        
        return {
            "strokes": all_strokes,
            "brush_names": all_brush_names,
            "stages_history": stages_history,
            "final_rendered": final_rendered,
        }
    
    def _optimize_stage(
        self,
        stage_idx: int,
        strokes: List[BrushStroke],
        brush_names: List[str],
        target_image: torch.Tensor,
        renderer: DifferentiableRenderer,
        loss_fn: CombinedLoss,
        config: StageConfig,
        progress_callback=None,
    ) -> Dict:
        """优化单个阶段"""
        
        # 收集所有可优化参数
        params = []
        for stroke in strokes:
            params.extend(stroke.parameters())
        
        optimizer = optim.Adam([
            {'params': [s.raw_control_points for s in strokes], 'lr': config.lr},
            {'params': [s.raw_width for s in strokes], 'lr': config.lr_width},
            {'params': [s.raw_color for s in strokes], 'lr': config.lr_color},
            {'params': [s.raw_opacity for s in strokes], 'lr': config.lr_color * 0.5},
        ])
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.num_iterations, eta_min=config.lr * 0.01
        )
        
        best_loss = float('inf')
        best_strokes = None
        loss_history = []
        no_improve_count = 0
        
        for iteration in range(config.num_iterations):
            optimizer.zero_grad()
            
            # 渲染当前笔画
            stroke_groups = self._group_strokes_by_brush(strokes, brush_names)
            rendered = renderer.render_multi_brush(stroke_groups)
            
            # 计算损失
            total_loss, loss_dict = loss_fn(rendered, target_image, strokes, stage=stage_idx)
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            loss_val = total_loss.item()
            loss_history.append(loss_val)
            
            # 记录最佳结果
            if loss_val < best_loss:
                best_loss = loss_val
                best_strokes = copy.deepcopy(strokes)
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # 自适应笔画重初始化：如果长时间没有改善，重初始化最差的笔画
            if no_improve_count > 0 and no_improve_count % 100 == 0 and iteration < config.num_iterations - 100:
                self._reinitialize_worst_strokes(
                    strokes, brush_names, target_image, renderer, num_to_reinit=max(1, len(strokes) // 10)
                )
                print(f"  [Iter {iteration}] Reinitialized {max(1, len(strokes) // 10)} worst strokes")
            
            # 打印进度
            if iteration % 50 == 0 or iteration == config.num_iterations - 1:
                lr = scheduler.get_last_lr()[0]
                print(f"  [Iter {iteration:4d}] Loss: {loss_val:.6f} "
                      f"(pixel: {loss_dict['pixel']:.4f}, perceptual: {loss_dict['perceptual']:.4f}, "
                      f"ssim: {loss_dict['ssim']:.4f}) lr: {lr:.6f}")
            
            # 保存中间结果
            if config.num_iterations > 100 and iteration % max(1, config.num_iterations // 10) == 0:
                self._save_iteration_result(stage_idx, iteration, rendered, target_image)
            
            # 进度回调
            if progress_callback is not None:
                try:
                    progress_callback(stage_idx, iteration, config.num_iterations, loss_dict)
                except Exception:
                    pass
        
        # 使用最佳结果
        if best_strokes is not None:
            for i, (stroke, best_stroke) in enumerate(zip(strokes, best_strokes)):
                stroke.raw_control_points.data.copy_(best_stroke.raw_control_points.data)
                stroke.raw_width.data.copy_(best_stroke.raw_width.data)
                stroke.raw_color.data.copy_(best_stroke.raw_color.data)
                stroke.raw_opacity.data.copy_(best_stroke.raw_opacity.data)
        
        return {
            "strokes": strokes,
            "brush_names": brush_names,
            "loss_history": loss_history,
            "best_loss": best_loss,
        }
    
    def _reinitialize_worst_strokes(
        self,
        strokes: List[BrushStroke],
        brush_names: List[str],
        target_image: torch.Tensor,
        renderer: DifferentiableRenderer,
        num_to_reinit: int = 3,
    ):
        """
        重初始化效果最差的笔画。
        
        通过移除每条笔画后观察损失变化来评估笔画的重要性，
        重要性最低的笔画被重新初始化到残差最大的区域。
        """
        if len(strokes) <= num_to_reinit:
            return
        
        # 渲染当前完整图像
        stroke_groups = self._group_strokes_by_brush(strokes, brush_names)
        full_rendered = renderer.render_multi_brush(stroke_groups)
        
        # 计算每条笔画的重要性（移除后损失增加越多越重要）
        importances = []
        for i, (stroke, brush_name) in enumerate(zip(strokes, brush_names)):
            # 渲染不包含该笔画的图像
            other_strokes = strokes[:i] + strokes[i+1:]
            other_brushes = brush_names[:i] + brush_names[i+1:]
            other_groups = self._group_strokes_by_brush(other_strokes, other_brushes)
            other_rendered = renderer.render_multi_brush(other_groups)
            
            # 重要性 = 移除后与目标的差异 - 保留时与目标的差异
            diff_with = (full_rendered - target_image).abs().mean().item()
            diff_without = (other_rendered - target_image).abs().mean().item()
            importance = diff_without - diff_with  # 正值表示该笔画有帮助
            importances.append(importance)
        
        # 找到最不重要的笔画
        importances_tensor = torch.tensor(importances)
        _, worst_indices = importances_tensor.sort()
        worst_indices = worst_indices[:num_to_reinit].tolist()
        
        # 计算残差注意力图
        residual = (target_image - full_rendered).abs().mean(dim=0)
        if residual.max() > 0:
            residual = residual / residual.max()
        
        # 重初始化最差的笔画到残差最大的区域
        from utils.init import sample_from_attention_map, compute_local_gradient_direction
        
        for idx in worst_indices:
            stroke = strokes[idx]
            
            # 从残差注意力图采样新位置
            positions = sample_from_attention_map(residual, 1, temperature=0.5)
            start_x, start_y = positions[0][0].item(), positions[0][1].item()
            
            # 计算新方向
            angle = compute_local_gradient_direction(target_image, (start_x, start_y))
            length = np.random.uniform(0.05, 0.2)
            
            # 重新初始化控制点
            n_cp = stroke.num_control_points
            new_points = torch.zeros(n_cp, 2, device=self.device)
            for j in range(n_cp):
                t = j / (n_cp - 1)
                offset_x = start_x + length * t * np.cos(angle) + np.random.randn() * 0.01
                offset_y = start_y + length * t * np.sin(angle) + np.random.randn() * 0.01
                new_points[j] = torch.tensor([offset_x, offset_y])
            
            new_points = torch.logit(torch.clamp(new_points, 0.01, 0.99))
            stroke.raw_control_points.data.copy_(new_points)
            
            # 重新采样颜色
            img_y = int(start_y * (target_image.shape[1] - 1))
            img_x = int(start_x * (target_image.shape[2] - 1))
            new_color = target_image[:, img_y, img_x].clone()
            new_color = torch.cat([new_color, torch.tensor([1.0], device=self.device)])
            stroke.raw_color.data.copy_(
                torch.logit(torch.clamp(new_color, 0.01, 0.99))
            )
    
    def _group_strokes_by_brush(
        self,
        strokes: List[BrushStroke],
        brush_names: List[str],
    ) -> List[Tuple[str, List[BrushStroke]]]:
        """按笔刷类型分组笔画"""
        groups = {}
        for stroke, brush_name in zip(strokes, brush_names):
            if brush_name not in groups:
                groups[brush_name] = []
            groups[brush_name].append(stroke)
        return list(groups.items())
    
    def _scale_strokes_to_canvas(
        self,
        strokes: List[BrushStroke],
        from_resolution: Tuple[int, int],
        to_canvas_size: Tuple[int, int],
    ) -> List[BrushStroke]:
        """将笔画坐标从一个分辨率缩放到另一个画布大小"""
        scale_x = to_canvas_size[0] / from_resolution[0]
        scale_y = to_canvas_size[1] / from_resolution[1]
        
        scaled = []
        for stroke in strokes:
            new_stroke = BrushStroke(
                num_control_points=stroke.num_control_points,
                canvas_size=to_canvas_size,
                init_width=stroke.width.item() * min(scale_x, scale_y),
                init_color=stroke.color.clone().detach(),
                init_opacity=stroke.opacity.item(),
                device=self.device,
            )
            # 缩放控制点
            cp = stroke.control_points.detach().clone()
            new_cp = cp.clone()
            new_cp[:, 0] = cp[:, 0]  # 归一化坐标不需要缩放
            new_cp[:, 1] = cp[:, 1]
            new_stroke.raw_control_points = nn.Parameter(
                torch.logit(torch.clamp(new_cp, 0.01, 0.99))
            )
            scaled.append(new_stroke)
        
        return scaled
    
    def _save_stage_result(
        self,
        stage_idx: int,
        rendered: torch.Tensor,
        strokes: List[BrushStroke],
        brush_names: List[str],
    ):
        """保存阶段结果"""
        stage_dir = os.path.join(self.output_dir, f"stage_{stage_idx + 1}")
        os.makedirs(stage_dir, exist_ok=True)
        
        save_image(rendered, os.path.join(stage_dir, "final_render.png"))
        
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
