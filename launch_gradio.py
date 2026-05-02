import gradio as gr
import numpy as np
from PIL import Image
from models.sfmm import init_sfmm_model, preprocess, analyze_sfmm_attention
from models.safe_df import init_safe_df_model, safe_df_predict
import torch
import os
import tempfile

# user_gradio_temp = os.path.expanduser("~/.gradio_temp")
tempfile.tempdir = './uploaded_data'

device = 'cuda'
load_8bit = torch.cuda.get_device_properties(0).total_mem < 16 * 1024**3 if torch.cuda.is_available() else False
sfmm_model = init_sfmm_model(device=device)
tokenizer, model, image_processor, context_len = init_safe_df_model(load_8bit=load_8bit)



def clear_outputs():
    """清空所有输出组件：文本置空、图片置None"""
    return (
        gr.update(value=""),          # 清空output_label（标签/文本）
        gr.update(value=None),       # 清空output_heatmap（图片组件）
        gr.update(value="")          # 清空output_explanation（解释文本）
    )
    
    
def preprocess_image_center_crop(pil_img, size=(224, 224)):
    # 1. 计算缩放比例，使较短的一边达到 224
    width, height = pil_img.size
    aspect_ratio = width / height
    
    if width > height:
        new_height = size[1]
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = size[0]
        new_height = int(new_width / aspect_ratio)
        
    # 2. 第一次等比例缩放
    pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 3. 计算中心裁剪坐标
    left = (new_width - size[0]) / 2
    top = (new_height - size[1]) / 2
    right = (new_width + size[0]) / 2
    bottom = (new_height + size[1]) / 2
    
    # 4. 执行裁剪
    return pil_img.crop((left, top, right, bottom))

# --- 模型推理函数 (请在此处接入你自己的模型) ---
def detect_forgery(input_img, question):
    """
    input_img: PIL Image
    question: str (用户选择的问题)
    """
    
    # image = PIL.Image.Image
    
    # SFMM
    input_img_sfmm = preprocess_image_center_crop(input_img, size=(224, 224))
    image = preprocess(input_img_sfmm).unsqueeze(0).to(device)
    with torch.inference_mode():
        sfmm_output = sfmm_model({'image':image, 'label':torch.tensor(1)}, inference=True)
    sfmm_output_image = analyze_sfmm_attention(input_img_sfmm, sfmm_output)
    sfmm_output_label = round(float(sfmm_output['prob'][0].detach().cpu()),3)
    
    
    # SAFE-DF
    input_img_safe_df = preprocess_image_center_crop(input_img, size=(336,336))
    safe_df_output = safe_df_predict(input_img_safe_df, question, tokenizer, model, image_processor, context_len)
    # safe_df_output = "这是一个模拟的解释性分析结果，实际结果请接入 SAFE-DF 模型进行推理。"
    
    # 输出：
    label = {'fake': sfmm_output_label, 'real':1-sfmm_output_label}
    mock_heatmap = sfmm_output_image
    explanation = safe_df_output
    
    return label, mock_heatmap, safe_df_output



# 准备 examples
real_example_list = [

        ['./examples/real_people2.png', "Does the image look real/fake?"],

        ["./examples/real_4.png", "Does the person's nose look real/fake?"],
]

fake_example_list = [

        ['./examples/nt2.png', "Does the image look real/fake?"],
        
        ['./examples/deepfakes1.png', "Does the person's eyes look real/fake?"],
        
        ['./examples/fs5.png', "Does the person's skin look real/fake?"],
        
    ]

# --- 构建 Gradio 界面 ---
with gr.Blocks(title="通用的人脸伪造检测系统") as demo:
    gr.Markdown("# 🛡️ 人脸伪造检测与可解释性分析系统")
    gr.Markdown("上传人脸图像并选择一个问题，模型将分析其真实性并给出可视化依据。")
    
    with gr.Row():
        # --- 左侧：输入层 ---
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="人脸图像", height=300, sources=["upload", "webcam", "clipboard"])
            
            question_dropdown = gr.Dropdown(
                choices=[
                    "Does the image look real/fake?",
                    "Does the person's nose look real/fake?",
                    "Does the person's skin look real/fake?",
                    "Does the person's mouth look real/fake?",
                    "Does the person's eyes look real/fake?"
                ],
                value="Does the image look real/fake?",  # 默认值
                label="检测问题"
            )
            
            submit_btn = gr.Button("开始分析", variant="primary")

        # --- 右侧：输出层 ---
        with gr.Column(scale=1):
            with gr.Group():
                # gr.Markdown("### 检测结果")
                output_label = gr.Label(label="SFMM 真伪判断")
                
                # gr.Markdown("### Attention Map (模型关注区域)")
                output_heatmap = gr.Image(label="SFMM 注意力热力图")
                
                # gr.Markdown("### 伪造原因解释")
                output_explanation = gr.Textbox(label="SAFE-DF 解释性分析", lines=4)
                
    gr.Examples(
        examples=real_example_list,          
        inputs=[input_img, question_dropdown], 
        outputs=[output_label, output_heatmap, output_explanation],  
        label="Real Detection Examples",      
        cache_examples=False        
        # run_on_click=True,             
        # cache_examples=True,           # 缓存示例结果，提升速度
        # example_title=["示例1", "示例2", "示例3"],  # 给每个示例命名
    )
    
    gr.Examples(
        examples=fake_example_list,          
        inputs=[input_img, question_dropdown], 
        outputs=[output_label, output_heatmap, output_explanation],  
        label="Fake Detection Examples",      
        cache_examples=False        
        # run_on_click=True,             
        # cache_examples=True,           # 缓存示例结果，提升速度
        # example_title=["示例1", "示例2", "示例3"],  # 给每个示例命名
    )

    # 绑定事件
    submit_btn.click(
        fn=clear_outputs,
        inputs=[],
        outputs=[output_label, output_heatmap, output_explanation]
    ).then(                              # 第二步：执行核心检测
        fn=detect_forgery,
        inputs=[input_img, question_dropdown],
        outputs=[output_label, output_heatmap, output_explanation]
    )

demo.queue()
demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

# demo.launch()