#!/bin/bash
# 本地环境搭建脚本（通用，支持各种 NVIDIA GPU）
# 显存 >= 16GB 自动 FP16，< 16GB 自动 8-bit 量化

# 创建虚拟环境
conda create -n forgery_detect python=3.10 -y
conda activate forgery_detect

# PyTorch (CUDA 12.1，兼容 12.x 驱动)
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# 核心依赖（锁定版本）
pip install transformers==4.37.2
pip install peft==0.10.0
pip install accelerate==0.28.0

# 8-bit 量化支持（显存 < 16GB 时自动启用）
pip install bitsandbytes>=0.41.0

# 其他依赖
pip install gradio
pip install sentencepiece
pip install protobuf
pip install scipy
pip install scikit-learn
pip install matplotlib
pip install pyyaml
pip install shortuuid
pip install safetensors
pip install pillow

echo "========================================="
echo "环境创建完成！"
echo "========================================="
echo ""
echo "运行方式："
echo "  conda activate forgery_detect"
echo "  cd <gradio_demo目录>"
echo "  export PYTHONPATH=\"./models:\$PYTHONPATH\""
echo "  python launch_gradio.py"
echo ""
echo "显存 >= 16GB: 自动 FP16 加载"
echo "显存 <  16GB: 自动 8-bit 量化加载"
