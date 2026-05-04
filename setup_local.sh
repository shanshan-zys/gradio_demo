#!/bin/bash
# 依赖安装脚本（需先手动激活 conda 环境后再运行）
# 用法：
#   conda create -n forgery_detect python=3.10 -y
#   conda activate forgery_detect
#   bash setup_local.sh

# PyTorch (CUDA 12.1，兼容 12.x 驱动)
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# 核心依赖（锁定版本）
pip install transformers==4.37.2
pip install peft==0.10.0
pip install accelerate==0.28.0

# 8-bit 量化支持（显存 < 16GB 时自动启用）
pip install bitsandbytes

# 其他依赖
pip install gradio sentencepiece protobuf scipy scikit-learn matplotlib pyyaml shortuuid safetensors pillow tensorboard facenet-pytorch

echo "========================================="
echo "依赖安装完成！"
echo "========================================="
echo ""
echo "运行方式："
echo "  export PYTHONPATH=\"./models:\$PYTHONPATH\""
echo "  python launch_gradio.py"
