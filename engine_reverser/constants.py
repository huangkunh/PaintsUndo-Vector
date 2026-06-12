"""
引擎参数常量 - 从 JS 引擎源码逆向提取

核心参数来源: enazo 笔刷工作方式.js
- d = {pointMax:64, tolerance:0.2, brushMinWidth:0.7, brushPower:0.07,
        brushNumberOfSegments:8, pressureBrushMinWidth:0.2, pressureBrushPower:0.2}
"""

# 画布尺寸（引擎固定）
CANVAS_WIDTH = 640
CANVAS_HEIGHT = 480

# 压感笔刷参数
PRESSURE_BRUSH_MIN_WIDTH = 0.2   # Z: 压感笔刷最小宽度比例
PRESSURE_BRUSH_POWER = 0.2       # H: 压感笔刷压感力度系数

# 无压感笔刷参数
BRUSH_MIN_WIDTH = 0.7            # b: 无压感笔刷最小宽度比例
BRUSH_POWER = 0.07               # p: 无压感笔刷力度系数
BRUSH_NUMBER_OF_SEGMENTS = 8     # M: Catmull-Rom 插值段数

# Catmull-Rom 样条参数
CATMULL_ROM_TENSION = 0.5        # Catmull-Rom 张力参数
CATMULL_ROM_SUBDIVISIONS = 6     # 压感笔刷插值细分数
CATMULL_ROM_SUBDIVISIONS_NOPRESS = 8  # 无压感笔刷插值细分数

# 铅笔参数 (brush 6)
PENCIL_WIDTH_FACTOR = 0.8        # lineWidth = width * (0.8 + pressure * ae)
PENCIL_PRESSURE_FACTOR = 0.07    # ae
PENCIL_ALPHA_POWER = 3           # alpha = (pressure/8)^3

# 水彩参数 (brush 9)
WATERCOLOR_WIDTH_FACTOR = 0.1    # lineWidth = width * (0.1 + pressure * Ae)
WATERCOLOR_PRESSURE_FACTOR = 0.2 # Ae

# 粗头笔刷参数 (brush 54)
THICK_HEAD_WIDTH_FACTOR = 0.1    # lineWidth = width * pressure * 0.1

# 混合模式
BLEND_SOURCE_OVER = "source-over"
BLEND_DESTINATION_OVER = "destination-over"
BLEND_DESTINATION_OUT = "destination-out"
BLEND_MULTIPLY = "multiply"
BLEND_SCREEN = "screen"
BLEND_DARKEN = "darken"

# 背景色索引
BACKGROUND_COLORS = [
    "#f8ecdb",  # 0: 暖白
    "#861010",  # 1: 深红
    "#080808",  # 2: 近黑
    "#956327",  # 3: 棕色
    "#93aeda",  # 4: 浅蓝
]

# 默认颜色
DEFAULT_COLORS = [
    "#222222", "#666666", "#ffffff", "#aaaaaa",
    "#d20000", "#1317f6", "#3b0c9b", "#f8c94e",
    "#CE8700", "#f3b2aa", "#008d26", "#25c9ff",
    "#ff7829", "#732d07", "#ff008f",
]

# 笔刷定义: {id: (name, step, has_pressure)}
BRUSHES = {
    0:  ("马克笔", 2, False),
    1:  ("原始（测试）", 2, False),
    5:  ("压感v3", 3, True),
    6:  ("铅笔", 3, True),
    9:  ("水彩", 3, True),
    10: ("虚线", 2, False),
    13: ("马克水性", 2, False),
    14: ("水性压感", 3, True),
    16: ("新压感尖头高性能", 3, True),
    18: ("区域", 2, False),
    19: ("区域水性", 2, False),
    28: ("喷笔v2", 3, True),
    34: ("蹭线", 3, True),
    36: ("排线", 3, True),
    39: ("渐变", 2, False),
    40: ("渐变背景", 2, False),
    54: ("粗头", 3, True),
    69: ("正态铅笔", 3, True),
}

# 常用笔刷 ID
BRUSH_MARKER = 0       # 马克笔 - 底色铺色
BRUSH_PRESSURE = 5     # 压感v3 - 主要刻画
BRUSH_PENCIL = 6       # 铅笔 - 线稿细节
BRUSH_WATERCOLOR = 9   # 水彩 - 柔和上色
BRUSH_AREA = 18        # 区域 - 大面积填色
BRUSH_THICK = 54       # 粗头 - 粗线条
