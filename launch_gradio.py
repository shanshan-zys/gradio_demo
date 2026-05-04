import gradio as gr
import numpy as np
from PIL import Image
from models.sfmm import init_sfmm_model, preprocess, analyze_sfmm_attention
from models.safe_df import init_safe_df_model, safe_df_predict
import torch
import os
import tempfile
from facenet_pytorch import MTCNN

tempfile.tempdir = './uploaded_data'

device = 'cuda'
load_8bit = torch.cuda.get_device_properties(0).total_memory < 16 * 1024**3 if torch.cuda.is_available() else False
sfmm_model = init_sfmm_model(device=device)
tokenizer, model, image_processor, context_len = init_safe_df_model(load_8bit=load_8bit)
mtcnn = MTCNN(keep_all=True, device=device if torch.cuda.is_available() else 'cpu')


def detect_and_crop_face(pil_img, margin=40, conf_threshold=0.98):
    boxes, probs = mtcnn.detect(pil_img)

    if boxes is None or len(boxes) == 0:
        return [], "未检测到人脸，请上传包含人脸的图片"

    filtered = [(box, prob) for box, prob in zip(boxes, probs) if prob >= conf_threshold]
    if len(filtered) == 0:
        return [], "未检测到人脸，请上传包含人脸的图片"

    boxes = [item[0] for item in filtered]
    w, h = pil_img.size
    all_faces = []
    for box in boxes:
        x1, y1, x2, y2 = box
        cx1 = max(0, int(x1 - margin))
        cy1 = max(0, int(y1 - margin))
        cx2 = min(w, int(x2 + margin))
        cy2 = min(h, int(y2 + margin))
        all_faces.append(pil_img.crop((cx1, cy1, cx2, cy2)))

    return all_faces, ""


def preprocess_image_center_crop(pil_img, size=(224, 224)):
    width, height = pil_img.size
    aspect_ratio = width / height

    if width > height:
        new_height = size[1]
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = size[0]
        new_height = int(new_width / aspect_ratio)

    pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    left = (new_width - size[0]) / 2
    top = (new_height - size[1]) / 2
    right = (new_width + size[0]) / 2
    bottom = (new_height + size[1]) / 2

    return pil_img.crop((left, top, right, bottom))


def detect_faces(input_img):
    if input_img is None:
        return "请先上传图片", [], [], -1, gr.update(interactive=False)

    all_faces, err = detect_and_crop_face(input_img)

    if len(all_faces) == 0:
        return err, [], [], -1, gr.update(interactive=False)

    thumbnail_size = (128, 128)
    thumbs = [f.resize(thumbnail_size, Image.Resampling.LANCZOS) for f in all_faces]

    n = len(all_faces)
    if n == 1:
        status = "检测到 1 张人脸，已自动选中"
        return status, thumbs, all_faces, 0, gr.update(interactive=True, variant="primary")
    else:
        status = f"检测到 {n} 张人脸，请点击选择要分析的人脸"
        return status, thumbs, all_faces, -1, gr.update(interactive=False)


def on_face_select(evt: gr.SelectData, face_list):
    idx = evt.index
    status = f"已选择第 {idx + 1} 张人脸"
    return idx, status, gr.update(interactive=True, variant="primary")


def analyze_forgery(question, face_list, selected_idx):
    if selected_idx < 0 or selected_idx >= len(face_list):
        return {}, None, ""

    face_img = face_list[selected_idx]

    # SFMM
    input_img_sfmm = preprocess_image_center_crop(face_img, size=(224, 224))
    image = preprocess(input_img_sfmm).unsqueeze(0).to(device)
    with torch.inference_mode():
        sfmm_output = sfmm_model({'image': image, 'label': torch.tensor(1)}, inference=True)
    sfmm_output_image = analyze_sfmm_attention(input_img_sfmm, sfmm_output)
    sfmm_output_label = round(float(sfmm_output['prob'][0].detach().cpu()), 3)

    # SAFE-DF
    input_img_safe_df = preprocess_image_center_crop(face_img, size=(336, 336))
    safe_df_output = safe_df_predict(input_img_safe_df, question, tokenizer, model, image_processor, context_len)

    label = {'fake': sfmm_output_label, 'real': 1 - sfmm_output_label}

    return label, sfmm_output_image, safe_df_output


def on_image_change():
    return (
        "",
        [],
        [],
        -1,
        gr.update(interactive=False, variant="secondary"),
        {},
        None,
        ""
    )


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
    gr.Markdown("# 人脸伪造检测与可解释性分析系统")
    gr.Markdown("上传人脸图像并选择一个问题，模型将分析其真实性并给出可视化依据。")

    face_list_state = gr.State([])
    selected_face_idx = gr.State(-1)

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

            with gr.Row():
                detect_btn = gr.Button("开始识别", variant="primary", scale=1)
                analyze_btn = gr.Button("开始分析", interactive=False, variant="secondary", scale=1)

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
        label="Real Detection Examples",
        cache_examples=False
    )

    gr.Examples(
        examples=fake_example_list,
        inputs=[input_img, question_dropdown],
        label="Fake Detection Examples",
        cache_examples=False
    )

    # 事件绑定
    detect_btn.click(
        fn=detect_faces,
        inputs=[input_img],
        outputs=[face_status, face_gallery, face_list_state, selected_face_idx, analyze_btn]
    )

    face_gallery.select(
        fn=on_face_select,
        inputs=[face_list_state],
        outputs=[selected_face_idx, face_status, analyze_btn]
    )

    analyze_btn.click(
        fn=analyze_forgery,
        inputs=[question_dropdown, face_list_state, selected_face_idx],
        outputs=[output_label, output_heatmap, output_explanation]
    )

    input_img.change(
        fn=on_image_change,
        inputs=[],
        outputs=[face_status, face_gallery, face_list_state, selected_face_idx, analyze_btn, output_label, output_heatmap, output_explanation]
    )

demo.queue()
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)), share=False)
