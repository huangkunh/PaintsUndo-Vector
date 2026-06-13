"""
语义图层系统 - 将"操作语义"显式化

核心思想：
- 人类画师不是"用笔画拟合图像"，而是"在语义图层上逐步构建画面"
- 每个图层有明确的语义：构图、线稿、底色、阴影、高光、调整
- 图层之间有严格的绘制顺序约束
- 同一图层内的笔画按特定策略排序

数据结构：
- PaintLayer: 语义图层（含名称、混合模式、笔画列表）
- SemanticStroke: 带语义标签的笔画（含图层归属、操作类型）
- PaintingPlan: 完整的绘画计划（含所有图层和执行顺序）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from enum import Enum, auto


class LayerType(Enum):
    """图层类型 - 对应人类画师的操作语义"""
    COMPOSITION = auto()    # 构图层：确定画面布局和大色块位置
    LINE_ART = auto()       # 线稿层：勾勒轮廓和结构线
    BASE_COLOR = auto()     # 底色层：大面积铺色，确定色调
    SHADOW = auto()         # 阴影层：添加暗部和投影
    MID_TONE = auto()       # 中间调层：过渡色和形体塑造
    HIGHLIGHT = auto()      # 高光层：亮部和反光
    DETAIL = auto()         # 细节层：纹理、边缘、小细节
    ADJUSTMENT = auto()     # 调整层：整体色彩/明暗调整
    ERASE = auto()          # 擦除层：修正错误


class BlendMode(Enum):
    """混合模式 - 对应 Canvas 引擎的 globalCompositeOperation"""
    SOURCE_OVER = "source-over"              # 正常绘制（前景覆盖背景）
    DESTINATION_OVER = "destination-over"    # 底层绘制（画在已有内容下面）
    DESTINATION_OUT = "destination-out"      # 擦除（移除已有内容）
    MULTIPLY = "multiply"                    # 正片叠底（阴影常用）
    SCREEN = "screen"                        # 滤色（高光常用）
    DARKEN = "darken"                        # 变暗


class OperationType(Enum):
    """操作类型 - 笔画的语义操作"""
    DRAW = auto()           # 正常绘制
    FILL = auto()           # 区域填充
    SKETCH = auto()         # 素描/线稿
    SHADE = auto()          # 阴影/暗部
    LIGHTEN = auto()        # 提亮/高光
    BLEND = auto()          # 颜色混合/过渡
    ERASE = auto()          # 擦除/修正
    ADJUST = auto()         # 整体调整
    TRANSFORM = auto()      # 翻转/旋转/缩放


# 图层类型 → 混合模式映射（人类画师的默认选择）
LAYER_BLEND_MAP = {
    LayerType.COMPOSITION: BlendMode.DESTINATION_OVER,
    LayerType.LINE_ART: BlendMode.SOURCE_OVER,
    LayerType.BASE_COLOR: BlendMode.DESTINATION_OVER,
    LayerType.SHADOW: BlendMode.MULTIPLY,
    LayerType.MID_TONE: BlendMode.SOURCE_OVER,
    LayerType.HIGHLIGHT: BlendMode.SCREEN,
    LayerType.DETAIL: BlendMode.SOURCE_OVER,
    LayerType.ADJUSTMENT: BlendMode.SOURCE_OVER,
    LayerType.ERASE: BlendMode.DESTINATION_OUT,
}

# 图层类型 → 默认透明度
LAYER_ALPHA_MAP = {
    LayerType.COMPOSITION: 0.6,
    LayerType.LINE_ART: 0.9,
    LayerType.BASE_COLOR: 0.85,
    LayerType.SHADOW: 0.5,
    LayerType.MID_TONE: 0.7,
    LayerType.HIGHLIGHT: 0.6,
    LayerType.DETAIL: 0.95,
    LayerType.ADJUSTMENT: 0.4,
    LayerType.ERASE: 1.0,
}

# 图层类型 → 默认笔刷
LAYER_BRUSH_MAP = {
    LayerType.COMPOSITION: "marker",
    LayerType.LINE_ART: "pencil",
    LayerType.BASE_COLOR: "watercolor",
    LayerType.SHADOW: "watercolor",
    LayerType.MID_TONE: "marker",
    LayerType.HIGHLIGHT: "airbrush",
    LayerType.DETAIL: "pressure",
    LayerType.ADJUSTMENT: "airbrush",
    LayerType.ERASE: "marker",
}

# 人类画师的图层执行顺序
HUMAN_LAYER_ORDER = [
    LayerType.COMPOSITION,
    LayerType.BASE_COLOR,
    LayerType.SHADOW,
    LayerType.LINE_ART,
    LayerType.MID_TONE,
    LayerType.HIGHLIGHT,
    LayerType.DETAIL,
    LayerType.ADJUSTMENT,
    LayerType.ERASE,
]


@dataclass
class SemanticStroke:
    """带语义标签的笔画"""
    stroke_data: Dict[str, Any]       # 笔画几何数据（控制点、宽度、颜色等）
    layer_type: LayerType             # 所属图层
    operation: OperationType          # 操作类型
    brush_name: str = "marker"        # 笔刷名称
    priority: float = 0.0            # 绘制优先级（越小越先画）
    confidence: float = 1.0          # 置信度（用于擦除/重画判断）
    is_correction: bool = False       # 是否为修正笔画
    replaced_stroke_idx: int = -1     # 被替换的笔画索引（-1表示无）


@dataclass
class PaintLayer:
    """语义图层"""
    layer_type: LayerType
    blend_mode: BlendMode = BlendMode.SOURCE_OVER
    alpha: float = 1.0
    strokes: List[SemanticStroke] = field(default_factory=list)
    is_visible: bool = True
    is_locked: bool = False          # 锁定后不可修改
    
    def add_stroke(self, stroke: SemanticStroke):
        self.strokes.append(stroke)
    
    def remove_stroke(self, idx: int):
        if 0 <= idx < len(self.strokes):
            self.strokes.pop(idx)
    
    def get_render_order(self) -> List[SemanticStroke]:
        """获取该图层内笔画的渲染顺序"""
        return sorted(self.strokes, key=lambda s: s.priority)


@dataclass
class PaintingPlan:
    """
    完整的绘画计划
    
    描述了从空白画布到完成作品的全部操作序列，
    包括图层结构、笔画数据、执行顺序和节奏控制。
    """
    layers: Dict[LayerType, PaintLayer] = field(default_factory=dict)
    execution_order: List[LayerType] = field(default_factory=list)
    target_image: Optional[torch.Tensor] = None
    canvas_size: Tuple[int, int] = (640, 480)
    
    def add_layer(self, layer_type: LayerType, blend_mode: Optional[BlendMode] = None,
                  alpha: Optional[float] = None):
        """添加一个语义图层"""
        if blend_mode is None:
            blend_mode = LAYER_BLEND_MAP.get(layer_type, BlendMode.SOURCE_OVER)
        if alpha is None:
            alpha = LAYER_ALPHA_MAP.get(layer_type, 1.0)
        
        self.layers[layer_type] = PaintLayer(
            layer_type=layer_type,
            blend_mode=blend_mode,
            alpha=alpha,
        )
        
        if layer_type not in self.execution_order:
            self.execution_order.append(layer_type)
    
    def add_stroke(self, layer_type: LayerType, stroke: SemanticStroke):
        """向指定图层添加笔画"""
        if layer_type not in self.layers:
            self.add_layer(layer_type)
        self.layers[layer_type].add_stroke(stroke)
    
    def get_all_strokes_ordered(self) -> List[SemanticStroke]:
        """获取按执行顺序排列的所有笔画"""
        all_strokes = []
        for layer_type in self.execution_order:
            if layer_type in self.layers:
                layer = self.layers[layer_type]
                if layer.is_visible:
                    all_strokes.extend(layer.get_render_order())
        return all_strokes
    
    def get_layer_at_step(self, step: int) -> LayerType:
        """获取指定步骤时正在绘制的图层"""
        cumulative = 0
        for layer_type in self.execution_order:
            if layer_type in self.layers:
                cumulative += len(self.layers[layer_type].strokes)
                if step < cumulative:
                    return layer_type
        return self.execution_order[-1] if self.execution_order else LayerType.ADJUSTMENT
    
    def total_strokes(self) -> int:
        return sum(len(l.strokes) for l in self.layers.values())
    
    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            "execution_order": [lt.name for lt in self.execution_order],
            "layers": {
                lt.name: {
                    "blend_mode": l.blend_mode.value,
                    "alpha": l.alpha,
                    "num_strokes": len(l.strokes),
                }
                for lt, l in self.layers.items()
            },
            "total_strokes": self.total_strokes(),
        }


class PaintingPlanBuilder:
    """
    绘画计划构建器 - 从目标图像自动生成绘画计划
    
    分析目标图像，确定：
    1. 需要哪些图层
    2. 每个图层有多少笔画
    3. 笔画的初始参数
    4. 执行顺序和节奏
    """
    
    def __init__(self, canvas_size: Tuple[int, int] = (640, 480), device: str = "cpu"):
        self.canvas_size = canvas_size
        self.device = device
    
    def build_plan(self, target: torch.Tensor) -> PaintingPlan:
        """
        从目标图像构建绘画计划
        
        分析步骤：
        1. 提取图像特征（边缘、颜色分布、亮度分布）
        2. 确定需要的图层
        3. 为每个图层分配笔画
        4. 设置执行顺序
        """
        plan = PaintingPlan(
            target_image=target,
            canvas_size=self.canvas_size,
        )
        
        C, H, W = target.shape
        
        # 分析图像特征
        luminance = 0.299 * target[0] + 0.587 * target[1] + 0.114 * target[2]
        edges = self._detect_edges(target)
        color_variance = self._color_variance(target)
        
        # 1. 构图层 - 总是存在
        plan.add_layer(LayerType.COMPOSITION, BlendMode.DESTINATION_OVER, 0.6)
        n_comp = max(8, int(30 * (1.0 - edges.mean().item())))
        for i in range(n_comp):
            stroke = SemanticStroke(
                stroke_data=self._init_composition_stroke(target, i, n_comp),
                layer_type=LayerType.COMPOSITION,
                operation=OperationType.FILL,
                brush_name="marker",
                priority=float(i),
            )
            plan.add_stroke(LayerType.COMPOSITION, stroke)
        
        # 2. 底色层 - 总是存在
        plan.add_layer(LayerType.BASE_COLOR, BlendMode.DESTINATION_OVER, 0.85)
        n_base = max(30, int(80 * color_variance.item()))
        for i in range(n_base):
            stroke = SemanticStroke(
                stroke_data=self._init_base_color_stroke(target, i, n_base),
                layer_type=LayerType.BASE_COLOR,
                operation=OperationType.FILL,
                brush_name="watercolor",
                priority=float(i),
            )
            plan.add_stroke(LayerType.BASE_COLOR, stroke)
        
        # 3. 阴影层 - 如果图像有明显的暗部
        shadow_ratio = (luminance < 0.3).float().mean().item()
        if shadow_ratio > 0.05:
            plan.add_layer(LayerType.SHADOW, BlendMode.MULTIPLY, 0.5)
            n_shadow = max(15, int(40 * shadow_ratio))
            for i in range(n_shadow):
                stroke = SemanticStroke(
                    stroke_data=self._init_shadow_stroke(target, luminance, i, n_shadow),
                    layer_type=LayerType.SHADOW,
                    operation=OperationType.SHADE,
                    brush_name="watercolor",
                    priority=float(i),
                )
                plan.add_stroke(LayerType.SHADOW, stroke)
        
        # 4. 线稿层 - 如果图像有明显的边缘
        edge_ratio = edges.mean().item()
        if edge_ratio > 0.02:
            plan.add_layer(LayerType.LINE_ART, BlendMode.SOURCE_OVER, 0.9)
            n_line = max(20, int(60 * edge_ratio))
            for i in range(n_line):
                stroke = SemanticStroke(
                    stroke_data=self._init_line_art_stroke(target, edges, i, n_line),
                    layer_type=LayerType.LINE_ART,
                    operation=OperationType.SKETCH,
                    brush_name="pencil",
                    priority=float(i),
                )
                plan.add_stroke(LayerType.LINE_ART, stroke)
        
        # 5. 中间调层
        plan.add_layer(LayerType.MID_TONE, BlendMode.SOURCE_OVER, 0.7)
        n_mid = max(20, int(50 * color_variance.item()))
        for i in range(n_mid):
            stroke = SemanticStroke(
                stroke_data=self._init_midtone_stroke(target, luminance, i, n_mid),
                layer_type=LayerType.MID_TONE,
                operation=OperationType.BLEND,
                brush_name="marker",
                priority=float(i),
            )
            plan.add_stroke(LayerType.MID_TONE, stroke)
        
        # 6. 高光层 - 如果图像有明显的亮部
        highlight_ratio = (luminance > 0.8).float().mean().item()
        if highlight_ratio > 0.03:
            plan.add_layer(LayerType.HIGHLIGHT, BlendMode.SCREEN, 0.6)
            n_highlight = max(10, int(30 * highlight_ratio))
            for i in range(n_highlight):
                stroke = SemanticStroke(
                    stroke_data=self._init_highlight_stroke(target, luminance, i, n_highlight),
                    layer_type=LayerType.HIGHLIGHT,
                    operation=OperationType.LIGHTEN,
                    brush_name="airbrush",
                    priority=float(i),
                )
                plan.add_stroke(LayerType.HIGHLIGHT, stroke)
        
        # 7. 细节层
        plan.add_layer(LayerType.DETAIL, BlendMode.SOURCE_OVER, 0.95)
        n_detail = max(30, int(80 * edge_ratio))
        for i in range(n_detail):
            stroke = SemanticStroke(
                stroke_data=self._init_detail_stroke(target, edges, i, n_detail),
                layer_type=LayerType.DETAIL,
                operation=OperationType.DRAW,
                brush_name="pressure",
                priority=float(i),
            )
            plan.add_stroke(LayerType.DETAIL, stroke)
        
        # 8. 调整层
        plan.add_layer(LayerType.ADJUSTMENT, BlendMode.SOURCE_OVER, 0.4)
        n_adjust = max(10, int(30 * (1.0 - color_variance.item())))
        for i in range(n_adjust):
            stroke = SemanticStroke(
                stroke_data=self._init_adjustment_stroke(target, i, n_adjust),
                layer_type=LayerType.ADJUSTMENT,
                operation=OperationType.ADJUST,
                brush_name="airbrush",
                priority=float(i),
            )
            plan.add_stroke(LayerType.ADJUSTMENT, stroke)
        
        # 设置执行顺序（人类画师逻辑）
        plan.execution_order = [lt for lt in HUMAN_LAYER_ORDER if lt in plan.layers]
        
        return plan
    
    def _detect_edges(self, image: torch.Tensor) -> torch.Tensor:
        """Sobel 边缘检测"""
        gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        gray = gray.unsqueeze(0).unsqueeze(0)
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)
        
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        edges = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8).squeeze()
        return edges / (edges.max() + 1e-8)
    
    def _color_variance(self, image: torch.Tensor) -> torch.Tensor:
        """颜色方差（衡量颜色丰富度）"""
        C, H, W = image.shape
        flat = image.reshape(3, -1)
        var = flat.var(dim=1).mean()
        return var.clamp(0, 1)
    
    def _sample_position(self, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[float, float]:
        """从图像中采样一个位置（偏向有内容的区域）"""
        C, H, W = target.shape
        if mask is not None:
            weights = mask.reshape(-1)
            weights = weights / (weights.sum() + 1e-8)
            idx = torch.multinomial(weights, 1).item()
            y, x = idx // W, idx % W
        else:
            y = np.random.randint(0, H)
            x = np.random.randint(0, W)
        return x / W, y / H
    
    def _init_composition_stroke(self, target, idx, total) -> Dict:
        """初始化构图笔画 - 大色块"""
        C, H, W = target.shape
        # 将画布分成网格，每个构图笔画覆盖一个区域
        cols = int(np.ceil(np.sqrt(total)))
        rows = int(np.ceil(total / cols))
        r, c = idx // cols, idx % cols
        cx = (c + 0.5) / cols
        cy = (r + 0.5) / rows
        
        # 采样该区域的平均颜色
        y1, y2 = int(r * H / rows), int((r + 1) * H / rows)
        x1, x2 = int(c * W / cols), int((c + 1) * W / cols)
        avg_color = target[:, y1:y2, x1:x2].mean(dim=(1, 2))
        
        return {
            "control_points": [[cx, cy], [cx + 0.1, cy + 0.05]],
            "width": 0.3,
            "color": avg_color.tolist() + [1.0],
            "opacity": 0.6,
        }
    
    def _init_base_color_stroke(self, target, idx, total) -> Dict:
        """初始化底色笔画 - 中等色块"""
        x, y = self._sample_position(target)
        C, H, W = target.shape
        px, py = int(x * W), int(y * H)
        color = target[:, py, px].tolist() + [1.0]
        
        return {
            "control_points": [[x, y], [x + 0.05, y + 0.03], [x + 0.08, y + 0.06]],
            "width": 0.15,
            "color": color,
            "opacity": 0.85,
        }
    
    def _init_shadow_stroke(self, target, luminance, idx, total) -> Dict:
        """初始化阴影笔画 - 暗部区域"""
        shadow_mask = (luminance < 0.3).float()
        x, y = self._sample_position(target, shadow_mask)
        C, H, W = target.shape
        px, py = int(x * W), int(y * H)
        color = target[:, py, px].tolist() + [1.0]
        # 阴影颜色偏暗
        color[:3] = [max(0, c * 0.7) for c in color[:3]]
        
        return {
            "control_points": [[x, y], [x + 0.04, y + 0.02], [x + 0.06, y + 0.05]],
            "width": 0.1,
            "color": color,
            "opacity": 0.5,
        }
    
    def _init_line_art_stroke(self, target, edges, idx, total) -> Dict:
        """初始化线稿笔画 - 沿边缘"""
        x, y = self._sample_position(target, edges)
        C, H, W = target.shape
        px, py = int(x * W), int(y * H)
        color = target[:, py, px].tolist() + [1.0]
        # 线稿颜色偏暗
        lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        if lum > 0.5:
            color[:3] = [c * 0.3 for c in color[:3]]
        
        return {
            "control_points": [[x, y], [x + 0.02, y + 0.01], [x + 0.04, y + 0.03], [x + 0.05, y + 0.02]],
            "width": 0.02,
            "color": color,
            "opacity": 0.9,
        }
    
    def _init_midtone_stroke(self, target, luminance, idx, total) -> Dict:
        """初始化中间调笔画"""
        mid_mask = ((luminance > 0.3) & (luminance < 0.7)).float()
        x, y = self._sample_position(target, mid_mask)
        C, H, W = target.shape
        px, py = int(x * W), int(y * H)
        color = target[:, py, px].tolist() + [1.0]
        
        return {
            "control_points": [[x, y], [x + 0.03, y + 0.02], [x + 0.05, y + 0.04]],
            "width": 0.08,
            "color": color,
            "opacity": 0.7,
        }
    
    def _init_highlight_stroke(self, target, luminance, idx, total) -> Dict:
        """初始化高光笔画 - 亮部区域"""
        highlight_mask = (luminance > 0.8).float()
        x, y = self._sample_position(target, highlight_mask)
        C, H, W = target.shape
        px, py = int(x * W), int(y * H)
        color = target[:, py, px].tolist() + [1.0]
        # 高光颜色偏亮
        color[:3] = [min(1, c * 1.2 + 0.1) for c in color[:3]]
        
        return {
            "control_points": [[x, y], [x + 0.02, y + 0.01]],
            "width": 0.05,
            "color": color,
            "opacity": 0.6,
        }
    
    def _init_detail_stroke(self, target, edges, idx, total) -> Dict:
        """初始化细节笔画"""
        x, y = self._sample_position(target, edges * 0.5 + 0.5)
        C, H, W = target.shape
        px, py = int(x * W), int(y * H)
        color = target[:, py, px].tolist() + [1.0]
        
        return {
            "control_points": [[x, y], [x + 0.01, y + 0.005], [x + 0.02, y + 0.01]],
            "width": 0.015,
            "color": color,
            "opacity": 0.95,
        }
    
    def _init_adjustment_stroke(self, target, idx, total) -> Dict:
        """初始化调整笔画"""
        x, y = self._sample_position(target)
        C, H, W = target.shape
        px, py = int(x * W), int(y * H)
        color = target[:, py, px].tolist() + [1.0]
        
        return {
            "control_points": [[x, y], [x + 0.04, y + 0.02]],
            "width": 0.2,
            "color": color,
            "opacity": 0.4,
        }
