import gradio as gr
import numpy as np
from PIL import Image
# from models.sfmm import init_sfmm_model, preprocess, analyze_sfmm_attention
# from models.safe_df import init_safe_df_model, safe_df_predict
import torch
import os
import tempfile

# user_gradio_temp = os.path.expanduser("~/.gradio_temp")
tempfile.tempdir = '/home/zhenghuang/forgery_detection/gradio_demo/uploaded_data'


def detect_forgery(input_img, question):
    """
    input_img: PIL Image
    question: str (用户选择的问题)
    """
    

    # 输出：
    label = {'fake': 1, 'real':0}
    mock_heatmap = './test.svg'
    explanation = 'test'
    
    return label, mock_heatmap, explanation


# --- 构建 Gradio 界面 ---
with gr.Blocks(title="通用的人脸伪造检测系统") as demo:
    gr.Markdown("# 🛡️ 人脸伪造检测与可解释性分析系统")
    gr.Markdown("上传人脸图像并选择一个问题，模型将分析其真实性并给出可视化依据。")
    
    with gr.Row():
        # --- 左侧：输入层 ---
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="上传人脸图像")
            
            question_dropdown = gr.Dropdown(
                choices=[
                    "Does the image look real/fake?",
                    "Does the person's nose look real/fake?",
                    "Does the person's skin look real/fake?",
                    "Does the person's mouth look real/fake?",
                    "Does the person's eyes look real/fake?"
                ],
                value="Does the image look real/fake?",  # 默认值
                label="选择检测问题"
            )
            
            submit_btn = gr.Button("开始分析", variant="primary")

        # --- 右侧：输出层 ---
        with gr.Column(scale=1):
            with gr.Group():
                # gr.Markdown("### 检测结果")
                output_label = gr.Label(label="真伪判断")
                
                # gr.Markdown("### Attention Map (模型关注区域)")
                output_heatmap = gr.Image(label="注意力热力图")
                
                # gr.Markdown("### 伪造原因解释")
                output_explanation = gr.Textbox(label="解释性分析", lines=4)

    # 绑定事件
    submit_btn.click(
        fn=detect_forgery,
        inputs=[input_img, question_dropdown],
        outputs=[output_label, output_heatmap, output_explanation]
    )

# demo.launch(share=True) 

demo.launch(share=True)