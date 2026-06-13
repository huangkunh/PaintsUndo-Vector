"""
Action Tokenizer - 将引擎 Action 序列离散化为 Token 序列

Tokenizer 设计:
- ["background", 0] → [ACT_BACKGROUND, VAL_0]
- ["colour", "#222222"] → [ACT_COLOUR, COLOR_222222]
- ["width", 10] → [ACT_WIDTH, VAL_10]
- ["alpha", 0.8] → [ACT_ALPHA, VAL_8]  (量化为 0-10)
- ["blend", "source-over"] → [ACT_BLEND, BLEND_SOURCE_OVER]
- ["line", [5, 100, 200, 3, 105, 202, 4]] → [ACT_LINE, BRUSH_5, X_100, Y_200, P_3, X_105, Y_202, P_4, END_LINE]

特殊 Token:
- PAD = 0
- END_SEQUENCE = 1
- ACT_BACKGROUND = 2
- ACT_COLOUR = 3
- ACT_COLOR = 4
- ACT_WIDTH = 5
- ACT_RADIUS = 6
- ACT_ALPHA = 7
- ACT_BLEND = 8
- ACT_LINE = 9
- END_LINE = 10
"""

import json
import re
from typing import List, Tuple, Dict, Optional
from engine_reverser.constants import BRUSHES, BACKGROUND_COLORS, DEFAULT_COLORS


# ========== Token ID 定义 ==========
PAD = 0
END_SEQUENCE = 1
ACT_BACKGROUND = 2
ACT_COLOUR = 3
ACT_COLOR = 4
ACT_WIDTH = 5
ACT_RADIUS = 6
ACT_ALPHA = 7
ACT_BLEND = 8
ACT_LINE = 9
END_LINE = 10

# 动作类型 Token 范围: 2-10
# 值 Token 范围: 100+

# 颜色 Token: 100 + color_hash (量化为 256 级)
COLOR_TOKEN_BASE = 100
COLOR_TOKEN_MAX = 100 + 512 * 512  # R*512 + G*16 + B (4bit per channel)

# 数值 Token: 100000 + value (量化)
VALUE_TOKEN_BASE = 100000
VALUE_TOKEN_MAX = 100000 + 1000  # 0-999

# 坐标 Token: 200000 + coordinate (0-639 for X, 0-479 for Y)
X_TOKEN_BASE = 200000
Y_TOKEN_BASE = 200000 + 640

# 压感 Token: 300000 + pressure (0-80, 即 0.0-8.0, 步长0.1)
PRESSURE_TOKEN_BASE = 300000

# 笔刷 ID Token: 400000 + brush_id
BRUSH_TOKEN_BASE = 400000

# 混合模式 Token: 500000 + blend_mode_id
BLEND_TOKEN_BASE = 500000
BLEND_MODES = {
    "source-over": 0,
    "destination-over": 1,
    "destination-out": 2,
    "multiply": 3,
    "screen": 4,
    "darken": 5,
    "lighten": 6,
    "overlay": 7,
}

# 背景色 Token: 600000 + bg_index
BG_TOKEN_BASE = 600000


class ActionTokenizer:
    """Action 序列 Tokenizer"""
    
    def encode_action(self, action: List) -> List[int]:
        """将单个 Action 编码为 Token 序列"""
        action_type = action[0]
        payload = action[1] if len(action) > 1 else None
        
        if action_type == "background":
            return [ACT_BACKGROUND, BG_TOKEN_BASE + int(payload)]
        
        elif action_type == "colour":
            if isinstance(payload, str) and payload.startswith("#"):
                color_id = self._color_to_id(payload)
                return [ACT_COLOUR, COLOR_TOKEN_BASE + color_id]
            else:
                return [ACT_COLOUR, COLOR_TOKEN_BASE + int(payload)]
        
        elif action_type == "color":
            if isinstance(payload, str) and payload.startswith("#"):
                color_id = self._color_to_id(payload)
                return [ACT_COLOR, COLOR_TOKEN_BASE + color_id]
            else:
                return [ACT_COLOR, COLOR_TOKEN_BASE + int(payload)]
        
        elif action_type == "width":
            val = int(float(payload) * 10)  # 量化到 0.1 精度
            return [ACT_WIDTH, VALUE_TOKEN_BASE + val]
        
        elif action_type == "radius":
            return [ACT_RADIUS, VALUE_TOKEN_BASE + int(payload)]
        
        elif action_type == "alpha":
            val = int(float(payload) * 10)  # 量化到 0.1 精度
            return [ACT_ALPHA, VALUE_TOKEN_BASE + val]
        
        elif action_type == "blend":
            blend_id = BLEND_MODES.get(payload, 0)
            return [ACT_BLEND, BLEND_TOKEN_BASE + blend_id]
        
        elif action_type == "line":
            line_data = payload
            brush_id = int(line_data[0])
            brush_info = BRUSHES.get(brush_id, ("未知", 2, False))
            step = brush_info[1]
            has_pressure = brush_info[2]
            
            tokens = [ACT_LINE, BRUSH_TOKEN_BASE + brush_id]
            
            coords = line_data[1:]
            if has_pressure:
                for i in range(0, len(coords) - 2, 3):
                    x, y, p = int(coords[i]), int(coords[i+1]), coords[i+2]
                    tokens.append(X_TOKEN_BASE + x)
                    tokens.append(Y_TOKEN_BASE + y)
                    p_quantized = int(p * 10)  # 0.1 精度
                    tokens.append(PRESSURE_TOKEN_BASE + p_quantized)
            else:
                for i in range(0, len(coords) - 1, 2):
                    x, y = int(coords[i]), int(coords[i+1])
                    tokens.append(X_TOKEN_BASE + x)
                    tokens.append(Y_TOKEN_BASE + y)
            
            tokens.append(END_LINE)
            return tokens
        
        return [PAD]
    
    def decode_tokens(self, tokens: List[int]) -> List[List]:
        """将 Token 序列解码为 Action 序列"""
        actions = []
        i = 0
        
        while i < len(tokens):
            tok = tokens[i]
            
            if tok == END_SEQUENCE:
                break
            elif tok == PAD:
                i += 1
                continue
            
            elif tok == ACT_BACKGROUND:
                if i + 1 < len(tokens):
                    bg_idx = tokens[i+1] - BG_TOKEN_BASE
                    actions.append(["background", bg_idx])
                    i += 2
                else:
                    i += 1
            
            elif tok == ACT_COLOUR:
                if i + 1 < len(tokens):
                    color_id = tokens[i+1] - COLOR_TOKEN_BASE
                    color_str = self._id_to_color(color_id)
                    actions.append(["colour", color_str])
                    i += 2
                else:
                    i += 1
            
            elif tok == ACT_COLOR:
                if i + 1 < len(tokens):
                    color_id = tokens[i+1] - COLOR_TOKEN_BASE
                    color_str = self._id_to_color(color_id)
                    actions.append(["color", color_str])
                    i += 2
                else:
                    i += 1
            
            elif tok == ACT_WIDTH:
                if i + 1 < len(tokens):
                    val = (tokens[i+1] - VALUE_TOKEN_BASE) / 10.0
                    actions.append(["width", val])
                    i += 2
                else:
                    i += 1
            
            elif tok == ACT_RADIUS:
                if i + 1 < len(tokens):
                    val = tokens[i+1] - VALUE_TOKEN_BASE
                    actions.append(["radius", val])
                    i += 2
                else:
                    i += 1
            
            elif tok == ACT_ALPHA:
                if i + 1 < len(tokens):
                    val = (tokens[i+1] - VALUE_TOKEN_BASE) / 10.0
                    actions.append(["alpha", val])
                    i += 2
                else:
                    i += 1
            
            elif tok == ACT_BLEND:
                if i + 1 < len(tokens):
                    blend_id = tokens[i+1] - BLEND_TOKEN_BASE
                    blend_str = {v: k for k, v in BLEND_MODES.items()}.get(blend_id, "source-over")
                    actions.append(["blend", blend_str])
                    i += 2
                else:
                    i += 1
            
            elif tok == ACT_LINE:
                # 解析 line action
                if i + 1 < len(tokens):
                    brush_id = tokens[i+1] - BRUSH_TOKEN_BASE
                    brush_info = BRUSHES.get(brush_id, ("未知", 2, False))
                    step = brush_info[1]
                    has_pressure = brush_info[2]
                    
                    line_data = [brush_id]
                    j = i + 2
                    
                    while j < len(tokens) and tokens[j] != END_LINE:
                        if has_pressure:
                            # 读取 x, y, p
                            if j + 2 < len(tokens):
                                x = tokens[j] - X_TOKEN_BASE if X_TOKEN_BASE <= tokens[j] < Y_TOKEN_BASE else 0
                                y = tokens[j+1] - Y_TOKEN_BASE if Y_TOKEN_BASE <= tokens[j+1] < PRESSURE_TOKEN_BASE else 0
                                p = (tokens[j+2] - PRESSURE_TOKEN_BASE) / 10.0 if PRESSURE_TOKEN_BASE <= tokens[j+2] < BRUSH_TOKEN_BASE else 0
                                line_data.extend([x, y, p])
                                j += 3
                            else:
                                break
                        else:
                            # 读取 x, y
                            if j + 1 < len(tokens):
                                x = tokens[j] - X_TOKEN_BASE if X_TOKEN_BASE <= tokens[j] < Y_TOKEN_BASE else 0
                                y = tokens[j+1] - Y_TOKEN_BASE if Y_TOKEN_BASE <= tokens[j+1] < PRESSURE_TOKEN_BASE else 0
                                line_data.extend([x, y])
                                j += 2
                            else:
                                break
                    
                    actions.append(["line", line_data])
                    i = j + 1  # skip END_LINE
                else:
                    i += 1
            
            else:
                i += 1
        
        return actions
    
    def encode_sequence(self, actions: List[List]) -> List[int]:
        """将完整 Action 序列编码为 Token 序列"""
        tokens = []
        for action in actions:
            tokens.extend(self.encode_action(action))
        tokens.append(END_SEQUENCE)
        return tokens
    
    def decode_sequence(self, tokens: List[int]) -> List[List]:
        """将 Token 序列解码为完整 Action 序列"""
        return self.decode_tokens(tokens)
    
    def _color_to_id(self, color_str: str) -> int:
        """将 #RRGGBB 颜色量化为 ID"""
        if not color_str.startswith("#") or len(color_str) != 7:
            return 0
        r = int(color_str[1:3], 16) >> 4  # 4bit
        g = int(color_str[3:5], 16) >> 4
        b = int(color_str[5:7], 16) >> 4
        return r * 256 + g * 16 + b  # 0-4095
    
    def _id_to_color(self, color_id: int) -> str:
        """将 ID 反量化为 #RRGGBB 颜色"""
        color_id = max(0, min(color_id, 4095))
        r = (color_id >> 8) & 0xF
        g = (color_id >> 4) & 0xF
        b = color_id & 0xF
        # 4bit → 8bit
        r = (r << 4) | r
        g = (g << 4) | g
        b = (b << 4) | b
        return f"#{r:02x}{g:02x}{b:02x}"
    
    @staticmethod
    def vocab_size() -> int:
        """词表大小"""
        return 500000  # 足够覆盖所有 Token 范围


def parse_log_file(log_path: str) -> List[List]:
    """解析引擎 JSON 日志文件"""
    with open(log_path, 'r') as f:
        actions = json.load(f)
    return actions


def extract_state_machine(actions: List[List]) -> List[Dict]:
    """
    从 Action 序列提取状态机轨迹
    
    返回每个 Action 执行后的状态快照
    """
    state = {
        "color": "#222222",
        "width": 4.0,
        "alpha": 1.0,
        "blend": "source-over",
        "background": "#f8ecdb",
    }
    
    states = []
    for action in actions:
        action_type = action[0]
        payload = action[1] if len(action) > 1 else None
        
        if action_type == "background":
            state["background"] = BACKGROUND_COLORS[payload] if isinstance(payload, int) and payload < len(BACKGROUND_COLORS) else "#f8ecdb"
        elif action_type in ("colour", "color"):
            if isinstance(payload, str):
                state["color"] = payload
            elif isinstance(payload, int) and payload < len(DEFAULT_COLORS):
                state["color"] = DEFAULT_COLORS[payload]
        elif action_type == "width":
            state["width"] = float(payload)
        elif action_type == "radius":
            radius_list = [2, 4, 10, 20]
            idx = min(int(payload), len(radius_list) - 1)
            state["width"] = float(radius_list[idx])
        elif action_type == "alpha":
            state["alpha"] = float(payload)
        elif action_type == "blend":
            state["blend"] = payload
        
        states.append(dict(state))
    
    return states
