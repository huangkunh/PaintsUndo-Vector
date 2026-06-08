# PaintsUndo-Vector

**基于可微渲染优化的矢量笔画生成工具** — 类似 Paints-Undo 的绘画过程还原，但输出的是干净的矢量笔画而非像素视频。

## 项目简介

PaintsUndo-Vector 采用**方案B（基于优化的高质量复刻）**实现，核心思路是：

1. **放弃视频扩散模型**，转而使用可微渲染 + 梯度优化
2. **采用文件笔刷工作方式**，将 enazo 绘画应用中的多种笔刷（马克笔、铅笔、水彩、压感笔等）作为可微渲染的基本图元
3. **多阶段渐进式优化**，模拟人类绘画过程：底色 → 形体刻画 → 细节线稿
4. **输出矢量笔画**，可直接导入 Illustrator / SVG 编辑器二次编辑

## 核心特性

- 🎨 **多种笔刷类型**：马克笔、铅笔、水彩、压感笔、喷笔、渐变等，源自 enazo 笔刷系统
- 🔄 **渐进式优化**：三阶段优化策略模拟人类绘画过程
- 📐 **矢量输出**：所有笔画均为贝塞尔曲线，可无限缩放
- 🎬 **回放系统**：支持绘画过程动态回放和 Undo 模拟
- 🖼️ **感知损失**：使用 LPIPS 感知损失 + 像素损失，确保语义对齐
- ⚡ **GPU 加速**：基于 PyTorch 的全流程 GPU 加速

## 技术架构

```
输入图像 → 笔刷参数初始化 → 可微渲染 → 损失计算 → 梯度优化 → 矢量笔画导出
                ↑                                              ↓
          多阶段策略 ←←←←←←←←←←←←←←←←←←←←←←←←←←←←← 绘画历史记录
```

### 三阶段优化策略

| 阶段 | 目标 | 笔画数 | 笔宽 | 分辨率 | 迭代次数 |
|------|------|--------|------|--------|----------|
| Stage 1: 铺底色 | 大面积色彩覆盖 | ~20条 | 20-50 | 256×256 | 500 |
| Stage 2: 形体刻画 | 轮廓和色块过渡 | ~100条 | 5-15 | 512×512 | 800 |
| Stage 3: 细节线稿 | 高频细节勾勒 | ~300条 | 1-3 | 1024×1024 | 1000+ |

## 安装

### 环境要求

- Python >= 3.9
- PyTorch >= 2.0.0（需 CUDA 支持）
- GPU: 推荐 8GB+ 显存

### 安装步骤

```bash
# 克隆项目
git clone https://github.com/your-username/PaintsUndo-Vector.git
cd PaintsUndo-Vector

# 安装依赖
pip install -e .

# 安装 pydiffvg（可微矢量渲染核心）
pip install pydiffvg
```

## 快速开始

### 命令行使用

```bash
# 基本用法：输入图片，生成矢量笔画
paints-undo-vector --input target.png --output output/

# 指定笔刷类型和优化参数
paints-undo-vector --input target.png --output output/ \
    --brush marker,pencil,watercolor \
    --stages 3 \
    --stage1-strokes 20 \
    --stage2-strokes 100 \
    --stage3-strokes 300

# 使用配置文件
paints-undo-vector --config configs/default.yaml --input target.png
```

### Python API 使用

```python
from core.optimizer import StrokeOptimizer
from brushes import MarkerBrush, PencilBrush, WatercolorBrush

# 创建优化器
optimizer = StrokeOptimizer(
    target_image="target.png",
    brushes=[MarkerBrush, PencilBrush, WatercolorBrush],
    canvas_size=(1024, 1024),
)

# 运行多阶段优化
history = optimizer.optimize()

# 导出 SVG
history.export_svg("output/painting_process.svg")

# 导出回放
history.export_replay("output/replay.json")
```

## 笔刷类型

基于 enazo 笔刷系统，支持以下笔刷类型：

| 笔刷 | ID | 说明 | 特点 |
|------|----|------|------|
| 马克笔 | 0 | 二次贝塞尔曲线 | 支持透明度叠加 |
| 铅笔 | 6 | 细线条 | 模拟铅笔质感 |
| 水彩 | 10 | 湿边扩散效果 | 模拟水彩晕染 |
| 压感笔v3 | 5 | 压力感应 | 笔宽随压力变化 |
| 新压感尖头 | 17 | 高性能压感 | 尖头效果，性能优化 |
| 喷笔v2 | 30 | 喷雾效果 | 模拟喷枪 |
| 渐变 | 39 | 线性渐变 | 从实色到透明 |
| 排线 | 37 | 平行线条 | 模拟素描排线 |
| 网点 | 32 | 半色调网点 | 漫画网点效果 |

## 项目结构

```
PaintsUndo-Vector/
├── brushes/           # 笔刷定义（源自 enazo 笔刷系统）
│   ├── __init__.py
│   ├── base.py        # 笔刷基类
│   ├── marker.py      # 马克笔
│   ├── pencil.py      # 铅笔
│   ├── watercolor.py  # 水彩笔
│   ├── pressure.py    # 压感笔系列
│   ├── airbrush.py    # 喷笔
│   ├── gradient.py    # 渐变笔刷
│   ├── hatching.py    # 排线笔刷
│   └── halftone.py    # 网点笔刷
├── core/              # 核心优化引擎
│   ├── __init__.py
│   ├── optimizer.py   # 多阶段优化器
│   ├── renderer.py    # 可微渲染器
│   ├── losses.py      # 损失函数
│   ├── scheduler.py   # 阶段调度器
│   └── cli.py         # 命令行入口
├── utils/             # 工具函数
│   ├── __init__.py
│   ├── image.py       # 图像处理
│   ├── init.py        # 笔画初始化策略
│   └── vis.py         # 可视化工具
├── export/            # 导出模块
│   ├── __init__.py
│   ├── svg_export.py  # SVG 导出
│   └── json_export.py # JSON 导出
├── replay/            # 回放模块
│   ├── __init__.py
│   └── player.py      # 绘画回放器
├── configs/           # 配置文件
│   └── default.yaml
├── examples/          # 示例脚本
│   └── basic_usage.py
├── pyproject.toml
└── README.md
```

## 技术细节

### 可微渲染管线

```
笔画参数 (points, width, color, opacity)
        ↓
    贝塞尔曲线构建
        ↓
    笔刷特效处理（根据笔刷类型）
        ↓
    光栅化渲染（pydiffvg）
        ↓
    画布合成（alpha blending）
        ↓
    损失计算 (L_pixel + λ·L_perceptual)
        ↓
    反向传播 → 更新笔画参数
```

### 损失函数

- **L_pixel**: L1 像素损失，确保颜色对齐
- **L_perceptual**: LPIPS 感知损失，确保语义对齐
- **L_length**: 笔画长度正则化，防止扭曲笔画
- **L_total = L_pixel + λ₁·L_perceptual + λ₂·L_length**

### 笔刷工作方式

本项目采用 enazo 绘画应用的笔刷工作方式，每种笔刷定义包含：

1. **渲染函数**：接收画布上下文、点序列、颜色、大小、透明度参数
2. **点收集策略**：不同笔刷有不同的采样方式（如马克笔使用二次贝塞尔曲线插值）
3. **特效参数**：如水彩的扩散半径、喷笔的喷雾密度等

## 致谢

- [DiffSketcher](https://github.com/ximinng/DiffSketcher) - 可微矢量渲染的先驱工作
- [pydiffvg](https://github.com/BachiLi/diffvg) - Adobe 开源的可微矢量渲染库
- [Paints-Undo](https://github.com/lllyasviel/Paints-UNDO) - 视频扩散模型绘画还原
- [enazo](https://enazo.cn) - 笔刷系统参考

## License

MIT License
