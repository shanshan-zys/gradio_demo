# 可解释人脸伪造检测交互演示系统

基于 Gradio 的交互式 Web 演示程序。用户可上传人脸图像或使用摄像头拍摄，系统通过 MTCNN 自动检测人脸，再结合 SFMM 和 SAFE-DF 两个模型输出真伪二分类判断、注意力热力图以及自然语言伪造解释。

## 系统架构

```
用户上传图片
    ↓
MTCNN 人脸检测（支持多人脸，用户可选择）
    ↓
┌───────────────────┬────────────────────────┐
│ SFMM 真伪判断      │ SAFE-DF 可解释性分析      │
│ (224×224)          │ (336×336)               │
│ · 真/假概率         │ · 自然语言伪造解释         │
│ · 注意力热力图       │                         │
└───────────────────┴────────────────────────┘
```

### 模型说明

**SFMM（Spatial-Frequency Multi-Modal Detector）**
- 基于 CLIP ViT-Base/16 的双分支检测器
- 空间分支处理 RGB 图像，频率分支对图像做 2D FFT 提取频域特征
- 输出真/假概率及空间-频率注意力热力图

**SAFE-DF（基于 LLaVA 的伪造解释模型）**
- 基于 LLaVA 1.5（LLaMA）架构，在 FF++ 数据集上微调
- 使用训练后的 Forgery-CLIP-L/336 作为视觉编码器
- 根据用户选择的问题生成自然语言伪造分析（如光照异常、纹理平滑等）
- 支持 8-bit 量化（GPU 显存 < 16GB 时自动启用）

## 目录结构

```
gradio_demo/
├── launch_gradio.py        # 主程序入口
├── run.sh                  # 启动脚本（含环境变量配置）
├── setup_local.sh          # 依赖安装脚本
├── launch_gradio_tool.py   # 备用调试启动脚本
├── model_predict.ipynb     # 独立模型推理调试 Notebook
├── examples/               # 演示用例图片
├── models/                 # 模型代码
│   ├── sfmm.py             # SFMM 模型加载与推理
│   ├── safe_df.py          # SAFE-DF 模型加载与推理
│   ├── dfb/                # SFMM 核心：CLIP 双分支检测器
│   └── llava/              # SAFE-DF 核心：LLaVA 架构
├── checkpoints/            # 模型权重（需手动下载）
│   ├── sfmm/               # SFMM 权重及配置
│   ├── clip-l-336/         # Forgery-CLIP 视觉编码器权重
│   ├── llava-1.5-finetune_merged_ff_unique_train/  # 微调后的 LLaVA 权重
│   └── models--openai--clip-vit-base-patch16/       # 基础 CLIP 模型
├── uploaded_data/          # 用户上传临时缓存
└── outputs/                # 运行输出文件
```

## 环境安装

### 1. 创建 conda 环境

```bash
conda create -n forgery_detect python=3.10 -y
conda activate forgery_detect
```

### 2. 安装依赖

```bash
bash setup_local.sh
```

主要依赖：PyTorch 2.4.1（CUDA 12.1）、Transformers 4.37.2、Gradio、facenet-pytorch 等。

GPU 显存 < 16GB 时会自动启用 8-bit 量化（需 bitsandbytes）。

### 3. 下载模型权重

将预训练权重放置到 `checkpoints/` 目录下。

下载链接：[ckpts.tar](https://pan.sjtu.edu.cn/web/share/406ef70159d37672ad5c0abb1f7e4fcc)，提取码：318l

## 启动

```bash
bash run.sh
```

默认在 `http://0.0.0.0:7860` 启动。如需更换端口：

```bash
GRADIO_SERVER_PORT=7861 bash run.sh
```

## 使用流程

1. 上传人脸图像或使用摄像头拍摄
2. 点击 **开始识别** 检测人脸
3. 如检测到多张人脸，点击缩略图选择要分析的人脸（单张人脸自动选中）
4. 选择检测问题（如 "Does the image look real/fake?"）
5. 点击 **开始分析** 获取结果：
   - SFMM 真伪概率及注意力热力图
   - SAFE-DF 自然语言伪造解释

## 分支说明

| 分支 | 说明 |
|------|------|
| `main` | 基础版本 |
| `base` | 添加 MTCNN 人脸检测预处理，一键检测分析 |
| `advanced` | 两步操作流程：先识别人脸再选择分析，支持多人脸交互选择 |
