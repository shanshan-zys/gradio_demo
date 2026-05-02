bash setup.sh
conda activate llava

cd gradio_demo
export PYTHONPATH="./models:$PYTHONPATH"
CUDA_VISIBLE_DEVICES=1 python launch_gradio.py