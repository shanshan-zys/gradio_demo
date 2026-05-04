import gradio as gr
import numpy as np
from PIL import Image
from models.sfmm import init_sfmm_model, preprocess, analyze_sfmm_attention
from models.safe_df import init_safe_df_model, safe_df_predict
import torch
import os
import tempfile
from facenet_pytorch import MTCNN

# user_gradio_temp = os.path.expanduser("~/.gradio_temp")
tempfile.tempdir = './uploaded_data'

device = 'cuda'
load_8bit = torch.cuda.get_device_properties(0).total_memory < 16 * 1024**3 if torch.cuda.is_available() else False
sfmm_model = init_sfmm_model(device=device)
tokenizer, model, image_processor, context_len = init_safe_df_model(load_8bit=load_8bit)
mtcnn = MTCNN(keep_all=True, device=device if torch.cuda.is_available() else 'cpu')



def clear_outputs():
    return (
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=None),
        gr.update(value=""),
        gr.update(value=None)
    )


def detect_and_crop_face(pil_img, margin=40, conf_threshold=0.98):
    boxes, probs = mtcnn.detect(pil_img)

    if boxes is None or len(boxes) == 0:
        return None, [], "未检测到人脸，请上传包含人脸的图片"

    # 过滤低置信度的检测结果
    filtered = [(box, prob) for box, prob in zip(boxes, probs) if prob >= conf_threshold]
    if len(filtered) == 0:
        return None, [], "未检测到人脸，请上传包含人脸的图片"

    boxes = [item[0] for item in filtered]
    w, h = pil_img.size
    all_faces = []
    areas = []
    for box in boxes:
        x1, y1, x2, y2 = box
        areas.append((x2 - x1) * (y2 - y1))
        cx1 = max(0, int(x1 - margin))
        cy1 = max(0, int(y1 - margin))
        cx2 = min(w, int(x2 + margin))
        cy2 = min(h, int(y2 + margin))
        all_faces.append(pil_img.crop((cx1, cy1, cx2, cy2)))

    best_idx = areas.index(max(areas))
    face_img = all_faces[best_idx]

    n_faces = len(boxes)
    if n_faces == 1:
        info = "检测到 1 张人脸"
    else:
        info = f"检测到 {n_faces} 张人脸，已选择最大的一张进行分析"

    return face_img, all_faces, info
    
    
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

    face_img, all_faces, face_status = detect_and_crop_face(input_img)
    if face_img is None:
        return {}, None, "", face_status, []

    # SFMM
    input_img_sfmm = preprocess_image_center_crop(face_img, size=(224, 224))
    image = preprocess(input_img_sfmm).unsqueeze(0).to(device)
    with torch.inference_mode():
        sfmm_output = sfmm_model({'image':image, 'label':torch.tensor(1)}, inference=True)
    sfmm_output_image = analyze_sfmm_attention(input_img_sfmm, sfmm_output)
    sfmm_output_label = round(float(sfmm_output['prob'][0].detach().cpu()),3)

    # SAFE-DF
    input_img_safe_df = preprocess_image_center_crop(face_img, size=(336,336))
    safe_df_output = safe_df_predict(input_img_safe_df, question, tokenizer, model, image_processor, context_len)

    label = {'fake': sfmm_output_label, 'real':1-sfmm_output_label}

    return label, sfmm_output_image, safe_df_output, face_status, all_faces



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
            input_img = gr.Image(type="pil", label="人脸图像", height=300, sources=["upload", "webcam"])

            question_dropdown = gr.Dropdown(
                choices=[
                    "Does the image look real/fake?",
                    "Does the person's nose look real/fake?",
                    "Does the person's skin look real/fake?",
                    "Does the person's mouth look real/fake?",
                    "Does the person's eyes look real/fake?"
                ],
                value="Does the image look real/fake?",
                label="检测问题"
            )

            submit_btn = gr.Button("开始分析", variant="primary")
            face_status = gr.Textbox(label="人脸检测状态", interactive=False)

        # --- 右侧：输出层 ---
        with gr.Column(scale=1):
            face_gallery = gr.Gallery(label="检测到的人脸", columns=4, rows=1, height=160, object_fit="cover", preview=False)
            with gr.Group():
                output_label = gr.Label(label="SFMM 真伪判断")
                output_heatmap = gr.Image(label="SFMM 注意力热力图")
                output_explanation = gr.Textbox(label="SAFE-DF 解释性分析", lines=4)
                
    gr.Examples(
        examples=real_example_list,
        inputs=[input_img, question_dropdown],
        outputs=[output_label, output_heatmap, output_explanation, face_status, face_gallery],
        label="Real Detection Examples",
        cache_examples=False
    )

    gr.Examples(
        examples=fake_example_list,
        inputs=[input_img, question_dropdown],
        outputs=[output_label, output_heatmap, output_explanation, face_status, face_gallery],
        label="Fake Detection Examples",
        cache_examples=False
    )

    # 绑定事件
    submit_btn.click(
        fn=clear_outputs,
        inputs=[],
        outputs=[output_label, output_explanation, output_heatmap, face_status, face_gallery]
    ).then(
        fn=detect_forgery,
        inputs=[input_img, question_dropdown],
        outputs=[output_label, output_heatmap, output_explanation, face_status, face_gallery]
    )

demo.queue()
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)), share=False)

# demo.launch()