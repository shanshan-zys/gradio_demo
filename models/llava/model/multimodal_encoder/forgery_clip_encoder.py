import torch
import torch.nn as nn

# from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from .forgery_clip import ForgeryCLIPVisionModel
from transformers import CLIPImageProcessor, CLIPVisionConfig
from collections import OrderedDict
import torch

def load_forgery_clip(pretrained_path=None):
    
    # if not pretrained_path:
    #     pretrained_path = '/home/zhenghuang/forgery_detection/LLaVA/pretrained_weights/raw_test_epoch5_lr0001/llava_clip_contrast_2025-10-06-17-28/epoch_ckpt/epoch1.pth'
    config = CLIPVisionConfig.from_pretrained('./checkpoints/clip-l-336')
    model = ForgeryCLIPVisionModel(config)
    pretrained_weight = torch.load(pretrained_path,map_location='cpu')
    new_state_dict = OrderedDict()
    for key, value in pretrained_weight.items():
        if "model.vision_model." in key:
            new_key = key.replace("model.vision_model.", "vision_model.")
            new_state_dict[new_key] = value
            
            
    print(f'---FORGERY CLIP ENCODER---\n loaded pretrain path == {pretrained_path}')
    # print(f'---clip config---\n {config}')
    # print(f'---model---\n{model}')
    # for n,p in model.named_parameters():
    #     print(f'{n}, {p.shape}')
        
        
    model.load_state_dict(new_state_dict, strict=True)
    print(f'Forgery clip from {pretrained_path} loaded!')
    return model

class ForgeryCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.mask = None

        self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained('./checkpoints/clip-l-336')
        self.vision_tower = load_forgery_clip(self.vision_tower_name)
        # self.vision_tower = ForgeryCLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
        
        if self.mask:
            if self.mask=='Hi':
                # -6
                rm = image_features.shape[1]-6
            if self.mask=='Hf':
                # -4
                rm = image_features.shape[1]-4
            if self.mask=='He':
                # -3
                rm = image_features.shape[1]-3
            image_features = torch.cat(
                [image_features[:, :rm, :],  # 前半部分：到索引 575 为止
                image_features[:, rm+1:, :]  # 后半部分：从索引 577 开始
                ], 
                dim=1  # 在第二个维度（seq_len）拼接
            )
        print(image_features.shape)
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


