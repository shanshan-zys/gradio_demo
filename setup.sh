# CUDA 12.4
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip
pip install torch torchvision
pip install transformers==4.37.2
pip install pillow peft==0.10.0
pip install accelerate==0.28.0
pip install sentencepiece
pip install protobuf wandb
pip install gradio shortuuid pyyaml scipy scikit-learn matplotlib
pip install flash-attn --no-build-isolation --no-cache-dir