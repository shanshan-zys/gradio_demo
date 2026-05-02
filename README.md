
# 可解释人脸伪造检测交互演示 Gradio Demo 

本项目是基于 Gradio 的交互式 Web 演示程序。通过直观的用户界面，用户可以上传人脸图像，系统将自动进行真伪鉴别，并基于 SFMM 和 SAFE-DF 输出伪造的二分类判断与自然语言伪造解释（如光照异常、纹理平滑等具体视觉证据）。

## 📁 目录结构与功能说明

```text
gradio_demo/
├── checkpoints/          # 模型权重存放目录 (运行前需确保权重已下载)
│   ├── clip-l-336                              # 训练后的 SAFE-DF 模型圈中
│   ├── llava-1.5-finetune_merged_ff_unique_train  # 在 FF++ 数据集上微调后的 LLaVA 权重（SAFE-DF）
│   ├── llava-v1.5-7b                           # 原始 LLaVA 基座模型权重
│   ├── models--openai--clip-vit-base-patch16   # 基础 CLIP 模型
│   └── sfmm                                    # SFMM 模型圈中
├── examples/             # 演示用例目录，存放用于一键测试的真假人脸图像
├── models/               # 模型核心代码库
├── outputs/              # 运行日志、分析结果或生成的中间图文对的输出目录
├── uploaded_data/        # 用户通过前端 Web 界面上传的临时缓存文件
├── frpc_linux_amd64_v0.2 # 用于生成公网可访问的 Gradio Share 共享链接
├── launch_gradio.py      # 主程序入口：启动标准版 Gradio Web UI
├── launch_gradio_tool.py # 备用启动脚本：包含工具箱或高级调试参数的 Web UI
├── model_predict.ipynb   # Jupyter Notebook：用于脱离 Web UI 的底层模型单步推理与调试
├── run.sh                # 核心启动脚本：包含环境变量配置的一键运行命令
└── setup.sh              # 环境初始化脚本：用于一键安装依赖包及配置环境

```

## 🛠️ 环境安装与配置

1. **环境初始化**
在项目根目录下，运行以下脚本自动安装所需的依赖库（包括 PyTorch, Gradio, Transformers 等）：
```bash
bash setup.sh

```


2. **准备模型权重**
请确保 `checkpoints` 目录下已正确放置项目所需的所有预训练权重文件。
预训练权重链接：ckpts.tar
链接: https://pan.sjtu.edu.cn/web/share/406ef70159d37672ad5c0abb1f7e4fcc, 提取码: 318l


3. **启动演示界面**


```bash
bash run.sh

```

