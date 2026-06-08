# PaintsUndo-Vector

**基于可微渲染优化的矢量笔画生成工具** — 类似 Paints-Undo 的绘画过程还原，但输出的是干净的矢量笔画而非像素视频。

## 项目简介

PaintsUndo-Vector 采用**方案B（基于优化的高质量复刻）**实现，核心思路是：

1. **放弃视频扩散模型**，转而使用可微渲染 + 梯度优化
2. **采用文件笔刷工作方式**，将 enazo 绘画应用中的多种笔刷（马克笔、铅笔、水彩、压感笔等）作为可微渲染的基本图元
3. **多阶段渐进式优化**，模拟人类绘画过程：底色 → 形体刻画 → 细节线稿
4. **输出矢量笔画**，可直接导入 Illustrator / SVG 编辑器二次编辑

## 核心特性

- 🎨 **9种笔刷类型**：马克笔、铅笔、水彩、压感笔、喷笔、渐变、排线、网点等，源自 enazo 笔刷系统
- 🔄 **渐进式优化**：三阶段优化策略模拟人类绘画过程
- 🧠 **注意力引导**：基于残差/边缘/显著性/颜色的注意力机制引导笔画放置
- 🖌️ **人类绘画模拟**：笔画排序、绘画节奏、颜色混合模拟真实画家
- 📐 **矢量输出**：所有笔画均为贝塞尔曲线，可无限缩放
- 🎬 **回放系统**：支持绘画过程动态回放、Undo/Redo、绘画叙述
- 🖼️ **感知损失**：LPIPS + SSIM + 颜色分布 + 像素损失，确保语义对齐
- ⚡ **GPU 加速**：基于 PyTorch 的全流程 GPU 加速

## 技术架构

```
输入图像 → 注意力图计算 → 笔画参数初始化 → 可微渲染 → 损失计算 → 梯度优化 → 矢量笔画导出
               ↑                                              ↓
          多阶段策略 ←←←←←←←←←←←←←←←←←←←←←←←←←←←←← 绘画历史记录
```

### 三阶段优化策略

| 阶段 | 目标 | 笔画数 | 笔宽 | 分辨率 | 迭代次数 | 主要笔刷 |
|------|------|--------|------|--------|----------|----------|
| Stage 1: 铺底色 | 大面积色彩覆盖 | ~20条 | 20-50 | 256×256 | 500 | 马克笔/水彩/喷笔 |
| Stage 2: 形体刻画 | 轮廓和色块过渡 | ~100条 | 5-15 | 512×512 | 800 | 压感笔/马克笔 |
| Stage 3: 细节线稿 | 高频细节勾勒 | ~300条 | 1-3 | 1024×1024 | 1000+ | 铅笔/排线 |

### 注意力引导机制

| 注意力类型 | 作用 | 底色权重 | 刻画权重 | 细节权重 |
|-----------|------|----------|----------|----------|
| 残差注意力 | 在差异最大处放置笔画 | 0.2 | 0.4 | 0.3 |
| 边缘注意力 | 在边缘处放置笔画 | 0.1 | 0.3 | 0.4 |
| 显著性注意力 | 在视觉显著处放置笔画 | 0.3 | 0.1 | 0.1 |
| 颜色注意力 | 在颜色差异处放置笔画 | 0.4 | 0.2 | 0.2 |

### 人类绘画模拟

- **笔画排序**：从背景到前景、从粗到细、从暗到亮
- **绘画节奏**：底色快速铺开 → 刻画放慢 → 细节最慢
- **颜色混合**：Porter-Duff over 操作模拟颜料叠加
- **绘画叙述**：模拟画家的内心独白

## 安装

### 环境要求

- Python >= 3.9
- PyTorch >= 2.0.0（推荐 CUDA 支持）
- GPU: 推荐 8GB+ 显存

### 安装步骤

```bash
# 克隆项目
git clone https://github.com/YOUR_USERNAME/PaintsUndo-Vector.git
cd PaintsUndo-Vector

# 安装依赖
pip install -e .
```

## 快速开始

### 命令行使用

```bash
# 基本用法：输入图片，生成矢量笔画
paints-undo-vector --input target.png --output output/

# 指定笔刷类型和优化参数
paints-undo-vector --input target.png --output output/ \
    --brush marker,pencil,watercolor

# 自定义配置文件
paints-undo-vector --input target.png --output output/ --config custom.yaml

# 仅运行特定阶段
paints-undo-vector --input target.png --output output/ --stage 2

# 指定 GPU
paints-undo-vector --input target.png --output result/ --device cuda:0
```

### Python API 使用

```python
import torch
from core import MultiStageOptimizer, DEFAULT_STAGE_CONFIGS
from utils.image import load_image

# 加载目标图像
target = load_image("photo.png", (640, 480), device="cuda")

# 创建优化器
optimizer = MultiStageOptimizer(
    canvas_size=(640, 480),
    device="cuda",
    output_dir="output/",
)

# 运行优化
result = optimizer.optimize(
    target_image=target,
    stage_configs=DEFAULT_STAGE_CONFIGS,
)

# 导出 SVG
from export import SVGExporter
exporter = SVGExporter(canvas_size=(640, 480))
exporter.export(result["strokes"], result["brush_names"], "output/result.svg")
```

## 项目结构

```
PaintsUndo-Vector/
├── brushes/             # 笔刷模块（源自 enazo）
│   ├── __init__.py
│   ├── base.py          # 笔刷基类 + BrushStroke 参数化
│   ├── marker.py        # 马克笔（二次贝塞尔曲线）
│   ├── pencil.py        # 铅笔（细线条+纹理噪声）
│   ├── watercolor.py    # 水彩（高斯模糊扩散）
│   ├── pressure.py      # 压感笔v3 + 新压感尖头高性能
│   ├── airbrush.py      # 喷笔v2（高斯喷雾）
│   ├── gradient.py      # 渐变（线性渐变填充）
│   ├── hatching.py      # 排线（平行线条填充）
│   └── halftone.py      # 网点（半色调网点效果）
├── core/                # 核心模块
│   ├── __init__.py
│   ├── renderer.py      # 可微渲染器（距离场光栅化）
│   ├── losses.py        # 损失函数（L1+LPIPS+SSIM+颜色分布）
│   ├── optimizer.py     # 多阶段优化器
│   ├── scheduler.py     # 阶段调度器
│   ├── attention.py     # 注意力引导（残差/边缘/显著性/颜色）
│   ├── painting_sim.py  # 人类绘画模拟（排序/节奏/混合）
│   └── cli.py           # 命令行入口
├── utils/               # 工具函数
│   ├── __init__.py
│   ├── image.py         # 图像处理
│   ├── init.py          # 笔画初始化策略（颜色聚类+边缘+显著性）
│   └── vis.py           # 可视化工具
├── export/              # 导出模块
│   ├── __init__.py
│   ├── svg_export.py    # SVG 导出（含动画+滤镜+变宽笔画）
│   └── json_export.py   # JSON 导出
├── replay/              # 回放模块
│   ├── __init__.py
│   └── player.py        # HTML 回放器（含叙述+节奏+快捷键）
├── configs/             # 配置文件
│   └── default.yaml
├── examples/            # 示例脚本
│   └── basic_usage.py
├── pyproject.toml
├── LICENSE
└── README.md
```

## 技术细节

### 可微渲染管线

```
笔画参数 (points, width, color, opacity)
        ↓
    De Casteljau 贝塞尔曲线求值
        ↓
    笔刷特效处理（根据笔刷类型）
        ↓
    距离场（SDF）计算
        ↓
    Smoothstep 抗锯齿光栅化
        ↓
    Alpha Blending 画布合成
        ↓
    损失计算 (L_pixel + λ₁·L_perceptual + λ₂·L_ssim + λ₃·L_color)
        ↓
    反向传播 → 更新笔画参数
```

### 损失函数

| 损失 | 作用 | 底色权重 | 刻画权重 | 细节权重 |
|------|------|----------|----------|----------|
| L_pixel | L1 像素损失 | 2.0 | 1.0 | 0.5 |
| L_perceptual | LPIPS 感知损失 | 5.0 | 10.0 | 15.0 |
| L_ssim | SSIM 结构损失 | 0.5 | 2.0 | 3.0 |
| L_color | 颜色分布损失 | 2.0 | 1.0 | 0.5 |
| L_length | 笔画长度正则化 | 0.01 | 0.005 | 0.003 |
| L_smooth | 笔画平滑正则化 | 0.005 | 0.003 | 0.001 |

### 笔刷工作方式

本项目采用 enazo 绘画应用的笔刷工作方式，每种笔刷定义包含：

1. **渲染函数**：接收画布上下文、点序列、颜色、大小、透明度参数
2. **点收集策略**：不同笔刷有不同的采样方式（如马克笔使用二次贝塞尔曲线插值）
3. **特效参数**：如水彩的扩散半径、喷笔的喷雾密度等

### 笔画初始化策略

| 阶段 | 初始化方式 | 颜色来源 | 位置来源 | 方向来源 |
|------|-----------|----------|----------|----------|
| 底色 | 颜色聚类 | K-Means主色调 | 颜色区域中心 | 随机 |
| 刻画 | 边缘引导 | 局部采样 | 边缘+显著性 | 梯度垂直方向 |
| 细节 | 高频引导 | 局部采样 | 高梯度区域 | 梯度方向 |

## 致谢

- [DiffSketcher](https://github.com/ximinng/DiffSketcher) - 可微矢量渲染的先驱工作
- [pydiffvg](https://github.com/BachiLi/diffvg) - Adobe 开源的可微矢量渲染库
- [Paints-Undo](https://github.com/lllyasviel/Paints-UNDO) - 视频扩散模型绘画还原
- [enazo](https://enazo.cn) - 笔刷系统参考

## License

MIT License
