# PaintsUndo-Vector

**基于可微渲染优化的矢量笔画生成工具** — 类似 Paints-Undo 的绘画过程还原，但输出的是干净的矢量笔画而非像素视频。

## 项目简介

PaintsUndo-Vector 采用**方案B（基于优化的高质量复刻）**实现，核心思路是：

1. **放弃视频扩散模型**，转而使用可微渲染 + 梯度优化
2. **采用文件笔刷工作方式**，将 glm 绘画应用中的多种笔刷（马克笔、铅笔、水彩、压感笔等）作为可微渲染的基本图元
3. **多阶段渐进式优化**，模拟人类绘画过程：底色 → 形体刻画 → 细节线稿
4. **输出矢量笔画**，可直接导入 Illustrator / SVG 编辑器二次编辑

## 核心特性

- 🎨 **9种笔刷类型**：马克笔、铅笔、水彩、压感笔、喷笔、渐变、排线、网点等
- 🔄 **渐进式优化**：三阶段优化策略模拟人类绘画过程
- 🧠 **注意力引导**：基于残差/边缘/显著性/颜色的注意力机制引导笔画放置
- 🖌️ **人类绘画模拟**：笔画排序、绘画节奏、手部微颤、颜色混合模拟真实画家
- 📐 **矢量输出**：所有笔画均为贝塞尔曲线，可无限缩放
- 🎬 **回放系统**：支持绘画过程动态回放、Undo/Redo、绘画叙述
- 🖼️ **感知损失**：VGG感知 + SSIM + 颜色分布 + 像素损失（纯PyTorch，无需额外依赖）
- ⚡ **GPU 加速**：基于 PyTorch 的全流程 GPU 加速
- 🌐 **Web UI**：Gradio 界面，无需命令行即可使用

## 快速开始

### 安装

```bash
git clone https://github.com/huangkunh/PaintsUndo-Vector.git
cd PaintsUndo-Vector
pip install -r requirements.txt
```

### Web UI（推荐）

```bash
pip install gradio
python app.py
```

然后打开 http://localhost:7860

### 命令行

```bash
# 基本用法
python -m core.cli --input target.png --output output/

# 指定笔刷和参数
python -m core.cli --input target.png --output output/ --brush marker,pencil,watercolor

# 自定义配置
python -m core.cli --input target.png --output output/ --config configs/default.yaml
```

### Python API

```python
import torch
from core.optimizer import MultiStageOptimizer, DEFAULT_STAGE_CONFIGS
from utils.image import load_image

target = load_image("photo.png", (640, 480), "cuda")
optimizer = MultiStageOptimizer(canvas_size=(640, 480), device="cuda", output_dir="output/")
result = optimizer.optimize(target, DEFAULT_STAGE_CONFIGS)
```

## 技术架构

```
输入图像 → 注意力图计算 → 笔画参数初始化 → 可微渲染 → 损失计算 → 梯度优化 → 矢量笔画导出
               ↑                                              ↓
          多阶段策略 ←←←←←←←←←←←←←←←←←←←←←←←←←←←←← 绘画历史记录
```

### 三阶段优化策略

| 阶段 | 目标 | 笔画数 | 笔宽 | 分辨率 | 迭代次数 | 主要笔刷 |
|------|------|--------|------|--------|----------|----------|
| Stage 1: 铺底色 | 大面积色彩覆盖 | ~32条 | 20-50 | 256×256 | 500 | 马克笔/水彩/喷笔 |
| Stage 2: 形体刻画 | 轮廓和色块过渡 | ~128条 | 5-15 | 512×512 | 800 | 压感笔/马克笔 |
| Stage 3: 细节线稿 | 高频细节勾勒 | ~512条 | 1-3 | 1024×1024 | 1000+ | 铅笔/排线 |

### 损失函数

| 损失 | 作用 | 底色权重 | 刻画权重 | 细节权重 |
|------|------|----------|----------|----------|
| L_pixel | L1 像素损失 | 2.0 | 1.0 | 0.5 |
| L_perceptual | VGG 感知损失 | 5.0 | 10.0 | 15.0 |
| L_ssim | SSIM 结构损失 | 0.5 | 2.0 | 3.0 |
| L_color | 颜色分布损失 | 2.0 | 1.0 | 0.5 |
| L_length | 笔画长度正则化 | 0.01 | 0.005 | 0.003 |
| L_smooth | 笔画平滑正则化 | 0.005 | 0.003 | 0.001 |

### 人类绘画模拟

- **笔画排序**：从背景到前景、从粗到细、从暗到亮
- **绘画节奏**：底色快速 → 刻画中速 → 细节慢速
- **手部微颤**：模拟人类手部的自然抖动
- **颜色混合**：模拟颜料的物理混合（over/multiply/average）
- **绘画叙述**：生成画家的内心独白文字

## 项目结构

```
PaintsUndo-Vector/
├── app.py              # Gradio Web UI
├── brushes/            # 笔刷实现
│   ├── base.py         # 笔刷基类 + BrushStroke 参数化
│   ├── marker.py       # 马克笔
│   ├── pencil.py       # 铅笔
│   ├── watercolor.py   # 水彩
│   ├── pressure.py     # 压感笔
│   ├── airbrush.py     # 喷笔
│   ├── gradient.py     # 渐变
│   ├── hatching.py     # 排线
│   └── halftone.py     # 网点
├── core/               # 核心模块
│   ├── renderer.py     # 可微渲染器
│   ├── optimizer.py    # 多阶段优化器
│   ├── losses.py       # 损失函数（纯PyTorch VGG感知损失）
│   ├── attention.py    # 注意力引导（纯PyTorch）
│   ├── painting_sim.py # 人类绘画模拟
│   ├── scheduler.py    # 阶段调度器
│   └── cli.py          # 命令行入口
├── utils/              # 工具函数
│   ├── image.py        # 图像处理
│   ├── init.py         # 笔画初始化（纯PyTorch）
│   └── vis.py          # 可视化工具
├── export/             # 导出模块
│   ├── svg_export.py   # SVG 导出
│   └── json_export.py  # JSON 导出
├── replay/             # 回放模块
│   └── player.py       # HTML 回放器
├── configs/            # 配置文件
│   └── default.yaml
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 致谢

- [DiffSketcher](https://github.com/ximinng/DiffSketcher) - 可微矢量渲染的先驱工作
- [pydiffvg](https://github.com/BachiLi/diffvg) - Adobe 开源的可微矢量渲染库
- [Paints-Undo](https://github.com/lllyasviel/Paints-UNDO) - 视频扩散模型绘画还原
- [glm](https://chatglm.cn) - 笔刷系统参考

## License

MIT License
