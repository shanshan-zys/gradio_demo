import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from .llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from .llava.conversation import conv_templates, SeparatorStyle
from .llava.model.builder import load_pretrained_model
from .llava.utils import disable_torch_init
from .llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from .llava.model.multimodal_encoder.forgery_clip_encoder import ForgeryCLIPVisionTower
from PIL import Image

def init_safe_df_model(load_8bit=False):
    model_path = './checkpoints/llava-1.5-finetune_merged_ff_unique_train'
    model_name = get_model_name_from_path(model_path)
    model_base = None
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, load_8bit=load_8bit)
    return tokenizer, model, image_processor, context_len


def safe_df_predict(input_image, qs, tokenizer, model, image_processor, context_len):
    device = model.device
    
    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    conv = conv_templates['llava_v1'].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

    image = input_image.convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0).half().to(device)

    with torch.inference_mode():
        # print(model.device, device, image_tensor.device, input_ids.device)
        model.model.vision_tower.to(device)
        model.model.vision_tower.mask = None
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=True,
            temperature=0.1,
            top_p=None,
            num_beams=1,
            # no_repeat_ngram_size=3,
            max_new_tokens=256,
            use_cache=True)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    return outputs
