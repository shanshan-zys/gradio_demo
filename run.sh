#!/bin/bash
# 首次运行先执行: bash setup_local.sh

export PYTHONPATH="./models:$PYTHONPATH"
CUDA_VISIBLE_DEVICES=0 python launch_gradio.py
