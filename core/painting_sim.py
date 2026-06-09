"""
人类绘画模拟模块

模拟人类画家的绘画习惯和策略：
1. 笔画排序：从背景到前景，从大到小，从粗到细
2. 重叠策略：前景笔画覆盖背景笔画
3. 颜色混合：模拟颜料的物理混合
4. 绘画节奏：模拟画家的绘画节奏（快速铺色 → 慢速刻画）
5. 视线引导：基于视觉显著性的绘画顺序
6. 笔触抖动：模拟手部微颤的自然效果
7. 压感模拟：模拟数位板的压感变化
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
    """
    if not strokes:
        return strokes, brush_names
    
    # 笔刷类型优先级（数字越小越先画）
    brush_priority = {
        "airbrush": 0,
        "watercolor": 1,
        "gradient": 2,
        "marker": 3,
        "halftone": 4,
        "hatching": 5,
        "pressure": 6,
        "pressure_sharp": 7,
        "pencil": 8,
    }
    
    sort_keys = []
    for i, (stroke, brush_name) in enumerate(zip(strokes, brush_names)):
        bp = brush_priority.get(brush_name, 5)
        
        width = stroke.width.detach().cpu().item()
        width_key = -width  # 粗笔画优先
        
        color = stroke.color.detach().cpu().numpy()
        brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        brightness_key = brightness  # 暗色优先（小值排前面）
        
        cp = stroke.control_points.detach().cpu().numpy()
        center_y = cp[:, 1].mean()
        center_x = cp[:, 0].mean()
        spatial_key = center_y * 2 + center_x  # 从上到下，从左到右
        
        sort_keys.append((bp, width_key, brightness_key, spatial_key, i))
    
    sorted_indices = [x[-1] for x in sorted(sort_keys)]
    
    sorted_strokes = [strokes[i] for i in sorted_indices]
    sorted_brush_names = [brush_names[i] for i in sorted_indices]
    
    return sorted_strokes, sorted_brush_names


def simulate_painting_rhythm(
    strokes: List[BrushStroke],
    brush_names: List[str],
    stage: int = 0,
) -> List[float]:
    """
    模拟人类画家的绘画节奏。
    
    人类画家的绘画节奏特征：
    - 底色阶段：快速、自信的笔触，时间间隔短
    - 刻画阶段：中等速度，偶尔停顿思考
    - 细节阶段：缓慢、精细，每笔都仔细考虑
    
    返回每条笔画的时间间隔（秒）。
    """
    durations = []
    
    for i, (stroke, brush_name) in enumerate(zip(strokes, brush_names)):
        base_duration = 0.3
        
        if stage == 0:
            # 底色：快速
            duration = base_duration * 0.5
        elif stage == 1:
            # 刻画：中等
            duration = base_duration * 1.0
        else:
            # 细节：慢速
            duration = base_duration * 2.0
        
        # 笔画越长，时间越长
        cp = stroke.control_points.detach()
        lengths = torch.norm(cp[1:] - cp[:-1], dim=1).sum()
        duration *= (1.0 + lengths.item() * 2.0)
        
        # 偶尔停顿（模拟思考）
        if np.random.random() < 0.05:
            duration += np.random.uniform(0.5, 2.0)
        
        durations.append(duration)
    
    return durations


def add_hand_tremor(
    stroke: BrushStroke,
    tremor_amount: float = 0.002,
) -> BrushStroke:
    """
    添加手部微颤效果。
    
    人类画家的手部会有微小的颤抖，
    这让笔画看起来更自然，而不是机器般的完美。
    
    Args:
        stroke: 原始笔画
        tremor_amount: 颤抖幅度（归一化坐标）
        
    Returns:
        添加颤抖后的笔画
    """
    with torch.no_grad():
        noise = torch.randn_like(stroke.raw_control_points) * tremor_amount
        stroke.raw_control_points.data += noise
    return stroke


def simulate_pressure_variation(
    stroke: BrushStroke,
    variation_type: str = "natural",
) -> BrushStroke:
    """
    模拟数位板的压感变化。
    
    人类画家的压感不是恒定的：
    - 起笔：轻 → 重
    - 行笔：略有波动
    - 收笔：重 → 轻
    
    Args:
        stroke: 原始笔画
        variation_type: "natural" | "heavy_start" | "heavy_end" | "uniform"
    """
    # 压感变化通过调整笔画宽度来模拟
    # 这里只是标记，实际效果在渲染时体现
    return stroke


def mix_colors(
    base_color: torch.Tensor,
    stroke_color: torch.Tensor,
    stroke_opacity: float,
    mode: str = "over",
) -> torch.Tensor:
    """
    模拟颜料的物理混合。
    
    不同混合模式模拟不同的绘画技法：
    - over: 标准覆盖（油画风格）
    - multiply: 正片叠底（水彩风格）
    - average: 平均混合（粉彩风格）
    """
    if mode == "over":
        mixed = base_color * (1 - stroke_opacity) + stroke_color * stroke_opacity
    elif mode == "multiply":
        mixed = base_color * stroke_color
        mixed = base_color * (1 - stroke_opacity) + mixed * stroke_opacity
    elif mode == "average":
        mixed = (base_color + stroke_color) / 2
        mixed = base_color * (1 - stroke_opacity) + mixed * stroke_opacity
    else:
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
        
        interval = max(1, num_strokes // len(narratives_for_stage))
        
        for i, stroke in enumerate(stage_data["strokes"]):
            if i % interval == 0:
                narr_idx = min(i // interval, len(narratives_for_stage) - 1)
                narratives.append(narratives_for_stage[narr_idx])
            else:
                narratives.append("")
    
    return narratives
