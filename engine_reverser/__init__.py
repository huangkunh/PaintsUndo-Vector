"""
引擎逆向笔画重建系统

基于特定渲染引擎的逆向笔画重建系统。
输入一张静态数字绘画图像，输出直接驱动特定Canvas笔刷引擎的Action序列。

模块:
- constants.py: 引擎参数常量
- renderer.py: 可微渲染器 (PyTorch)
- tokenizer.py: Action Tokenizer
- reverser.py: 逆向笔画重建器
- inference.py: 推理脚本
- validate.py: 验证脚本
- model/transformer.py: Transformer 模型 (GPT)
"""

from engine_reverser.constants import *
from engine_reverser.renderer import DifferentiableCanvasRenderer
from engine_reverser.tokenizer import ActionTokenizer
from engine_reverser.reverser import StrokeReverser
