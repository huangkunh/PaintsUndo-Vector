"""
人类绘画模拟模块

模拟人类画家的绘画习惯和策略：
1. 笔画排序：从背景到前景，从大到小，从粗到细
2. 重叠策略：前景笔画覆盖背景笔画
3. 颜色混合：模拟颜料的物理混合
4. 绘画节奏：模拟画家的绘画节奏（快速铺色 → 慢速刻画）
5. 视线引导：基于视觉显著性的绘画顺序

这些策略让生成的笔画序列更接近人类画家的绘画过程，
而不仅仅是优化结果的简单排列。
"""

from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from brushes.base import BrushStroke


def sort_strokes_human_like(
    strokes: List[BrushStroke],
    brush_names: List[str],
    target_image: Optional[torch.Tensor] = None,
) -> Tuple[List[BrushStroke], List[str]]:
    """
    按照人类绘画习惯排序笔画。
    
    排序规则（按优先级）：
    1. 笔刷类型：水彩/马克笔（底色）→ 压感笔（刻画）→ 铅笔（细节）
    2. 笔画宽度：粗笔画优先（先铺大色块）
    3. 颜色亮度：暗色优先（先画暗部，再画亮部）
    4. 空间位置：从上到下，从左到右（模拟画家的视线移动）
    5. 显著性：非显著区域优先（先画背景，再画前景）
    
    Args:
        strokes: 笔画列表
        brush_names: 对应的笔刷名称列表
        target_image: 目标图像（可选，用于显著性排序）
        
    Returns:
        排序后的 (strokes, brush_names)
    """
    if not strokes:
        return strokes, brush_names
    
    # 笔刷类型优先级（数字越小越先画）
    brush_priority = {
        "airbrush": 0,      # 喷笔最先（大面积底色）
        "watercolor": 1,    # 水彩其次
        "gradient": 2,      # 渐变
        "marker": 3,        # 马克笔
        "halftone": 4,      # 网点
        "hatching": 5,      # 排线
        "pressure": 6,      # 压感笔
        "pressure_sharp": 7, # 尖头压感笔
        "pencil": 8,        # 铅笔最后（细节）
    }
    
    # 计算每个笔画的排序键
    sort_keys = []
    for i, (stroke, brush_name) in enumerate(zip(strokes, brush_names)):
        # 1. 笔刷类型优先级
        bp = brush_priority.get(brush_name, 5)
        
        # 2. 笔画宽度（粗的先画）
        width = stroke.width.detach().cpu().item()
        width_score = -width  # 负号使粗笔画排在前面
        
        # 3. 颜色亮度（暗的先画）
        color = stroke.color.detach().cpu().numpy()
        brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        brightness_score = -brightness  # 负号使暗色排在前面
        
        # 4. 空间位置（从上到下，从左到右）
        cp = stroke.control_points.detach().cpu().numpy()
        center_y = cp[:, 1].mean()
        center_x = cp[:, 0].mean()
        spatial_score = center_y * 2 + center_x  # 上方和左方优先
        
        # 综合排序键
        key = (bp, width_score, brightness_score, spatial_score, i)
        sort_keys.append(key)
    
    # 排序
    sorted_indices = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])
    
    sorted_strokes = [strokes[i] for i in sorted_indices]
    sorted_brush_names = [brush_names[i] for i in sorted_indices]
    
    return sorted_strokes, sorted_brush_names


def compute_stroke_overlap(
    stroke1: BrushStroke,
    stroke2: BrushStroke,
    canvas_size: Tuple[int, int] = (640, 480),
) -> float:
    """
    计算两条笔画的空间重叠度。
    
    用于确定笔画的覆盖关系：
    重叠度高的笔画应该按特定顺序绘制。
    """
    cp1 = stroke1.control_points.detach().cpu().numpy()
    cp2 = stroke2.control_points.detach().cpu().numpy()
    
    # 计算两条笔画的包围盒
    min1_x, min1_y = cp1.min(axis=0)
    max1_x, max1_y = cp1.max(axis=0)
    min2_x, min2_y = cp2.min(axis=0)
    max2_x, max2_y = cp2.max(axis=0)
    
    # 考虑笔画宽度
    w1 = stroke1.width.detach().cpu().item() / canvas_size[0]
    w2 = stroke2.width.detach().cpu().item() / canvas_size[0]
    
    min1_x -= w1 / 2
    max1_x += w1 / 2
    min1_y -= w1 / 2
    max1_y += w1 / 2
    min2_x -= w2 / 2
    max2_x += w2 / 2
    min2_y -= w2 / 2
    max2_y += w2 / 2
    
    # 计算IoU
    inter_min_x = max(min1_x, min2_x)
    inter_min_y = max(min1_y, min2_y)
    inter_max_x = min(max1_x, max2_x)
    inter_max_y = min(max1_y, max2_y)
    
    if inter_max_x <= inter_min_x or inter_max_y <= inter_min_y:
        return 0.0
    
    inter_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
    area1 = (max1_x - min1_x) * (max1_y - min1_y)
    area2 = (max2_x - min2_x) * (max2_y - min2_y)
    union_area = area1 + area2 - inter_area
    
    return inter_area / (union_area + 1e-8)


def simulate_painting_rhythm(
    strokes: List[BrushStroke],
    brush_names: List[str],
) -> List[Dict]:
    """
    模拟画家的绘画节奏。
    
    为每条笔画分配时间戳和持续时间，
    模拟人类画家的绘画节奏：
    - 底色阶段：快速、连续的笔画
    - 刻画阶段：中等速度、有停顿
    - 细节阶段：缓慢、精细的笔画
    
    Returns:
        每条笔画的时间信息列表
    """
    brush_rhythm = {
        "airbrush": {"duration": 0.8, "pause": 0.1},
        "watercolor": {"duration": 0.6, "pause": 0.2},
        "gradient": {"duration": 0.5, "pause": 0.1},
        "marker": {"duration": 0.4, "pause": 0.15},
        "halftone": {"duration": 0.3, "pause": 0.1},
        "hatching": {"duration": 0.3, "pause": 0.1},
        "pressure": {"duration": 0.5, "pause": 0.2},
        "pressure_sharp": {"duration": 0.4, "pause": 0.15},
        "pencil": {"duration": 0.3, "pause": 0.1},
    }
    
    timeline = []
    current_time = 0.0
    
    for i, (stroke, brush_name) in enumerate(zip(strokes, brush_names)):
        rhythm = brush_rhythm.get(brush_name, {"duration": 0.4, "pause": 0.15})
        
        # 笔画长度影响持续时间
        stroke_length = stroke.get_length().item()
        duration = rhythm["duration"] * (0.5 + stroke_length)
        
        # 偶尔添加较长的停顿（模拟画家思考）
        pause = rhythm["pause"]
        if np.random.random() < 0.1:  # 10% 概率长停顿
            pause += 0.5
        
        timeline.append({
            "stroke_index": i,
            "start_time": current_time,
            "duration": duration,
            "end_time": current_time + duration,
            "brush_name": brush_name,
        })
        
        current_time += duration + pause
    
    return timeline


def compute_color_mixing(
    base_color: torch.Tensor,
    stroke_color: torch.Tensor,
    stroke_opacity: float,
    mixing_mode: str = "subtractive",
) -> torch.Tensor:
    """
    模拟颜料的物理混合。
    
    人类画家在绘画时，颜料会以物理方式混合：
    - 减色混合（subtractive）：像水彩一样，颜色越混越暗
    - 加色混合（additive）：像光一样，颜色越混越亮
    - 平均混合（average）：简单的颜色平均
    
    Args:
        base_color: 基础颜色 [3]
        stroke_color: 笔画颜色 [3]
        stroke_opacity: 笔画透明度
        mixing_mode: 混合模式
        
    Returns:
        混合后的颜色 [3]
    """
    if mixing_mode == "subtractive":
        # 减色混合：模拟水彩/油画颜料
        # CMYK 模型：颜色越混越暗
        mixed = base_color * (1 - stroke_opacity) + stroke_color * stroke_opacity
        # 轻微的减色效果
        darkening = 1.0 - 0.05 * stroke_opacity
        mixed = mixed * darkening
        
    elif mixing_mode == "additive":
        # 加色混合：模拟光
        mixed = base_color + stroke_color * stroke_opacity
        mixed = torch.clamp(mixed, 0, 1)
        
    else:  # average
        # 平均混合
        mixed = base_color * (1 - stroke_opacity) + stroke_color * stroke_opacity
    
    return torch.clamp(mixed, 0, 1)


def generate_painting_narrative(
    stages_history: List[Dict],
) -> List[str]:
    """
    生成绘画过程的文字叙述。
    
    模拟画家在绘画过程中的内心独白，
    用于回放系统中的文字提示。
    """
    narratives = []
    
    stage_narratives = {
        0: [
            "先铺一层底色...",
            "用大号画笔覆盖主要色块...",
            "确定画面的整体色调...",
            "底色要大胆，不要犹豫...",
        ],
        1: [
            "开始勾勒形体轮廓...",
            "注意明暗交界线...",
            "用中等画笔刻画色块过渡...",
            "调整形体比例...",
            "加深暗部，提亮亮部...",
        ],
        2: [
            "添加细节线条...",
            "用细铅笔勾勒纹理...",
            "注意边缘的虚实变化...",
            "最后调整细节...",
            "签名完成！",
        ],
    }
    
    for stage_idx, stage_data in enumerate(stages_history):
        num_strokes = len(stage_data["strokes"])
        narratives_for_stage = stage_narratives.get(stage_idx, ["继续绘画..."])
        
        # 每隔一定数量的笔画插入一条叙述
        interval = max(1, num_strokes // len(narratives_for_stage))
        
        for i, stroke in enumerate(stage_data["strokes"]):
            if i % interval == 0:
                narr_idx = min(i // interval, len(narratives_for_stage) - 1)
                narratives.append(narratives_for_stage[narr_idx])
            else:
                narratives.append("")
    
    return narratives
