'''

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. get_test_metrics: Testing-metrics-computation
9. forward: Forward-propagation

Reference:
@inproceedings{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  booktitle={International conference on machine learning},
  pages={8748--8763},
  year={2021},
  organization={PMLR}
}
'''

import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from .clip_df import CLIP_DF_Model, CLIPVisionTransformer_multi0828


from transformers import AutoProcessor, CLIPModel, ViTModel, ViTConfig

import copy
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig

logger = logging.getLogger(__name__)



class ClipMultiDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone,_ = self.build_backbone(config)
        self.backbone = CLIPVisionTransformer_multi0828(self.backbone, config['model_config'])

        num_forgery_types = 7
        self.freq_head = nn.Linear(768, 2)
        self.freq_spe_head = nn.Linear(768, num_forgery_types)

        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        
    def build_backbone(self, config):
        # prepare the backbone
        backbone, freq_backbone = get_clip_visual(config, model_name="large")
        return backbone, freq_backbone

    
    def features(self, data_dict: dict) -> torch.tensor:
        outputs = self.backbone(data_dict['image'],output_attentions=True)
        # outputs = self.backbone(data_dict['image'])
        # print(outputs)
        freq_feat = outputs['pooler_output']
        attns = outputs['attentions']
        # print(len(attns), data_dict['image'].shape)

        return {'freq_cls_feat': freq_feat[:,0,:], 'freq_spe_feat': freq_feat[:,1,:], 'attentions': attns}

    def classifier(self, features: torch.tensor) -> torch.tensor:

        freq_fake = self.freq_head(features['freq_cls_feat'])
        freq_spe_fake = self.freq_spe_head(features['freq_spe_feat'])

        return {'freq_cls': freq_fake, 'freq_spe_cls': freq_spe_fake}
    
    
    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        features = self.features(data_dict)
        # get the prediction by classifier
        pred_res = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred_res['freq_cls'], dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred_res['freq_cls'], 'prob': prob, 'feat': features['freq_cls_feat']}

        pred_dict = {
            'cls': pred_res['freq_cls'],
            'prob': torch.softmax(pred_res['freq_cls'], dim=1)[:, 1],
            'feat': features['freq_cls_feat'],
            'freq_spe_cls': pred_res['freq_spe_cls'],
            'freq_spe_prob': torch.softmax(pred_res['freq_spe_cls'], dim=1)[:, 1],
            'cls_feature': features['freq_cls_feat'],
            'spe_feature': features['freq_spe_feat'],
            'attentions': features['attentions']
        }

        return pred_dict


def get_clip_visual(config, model_name = "base"):
    # processor = AutoProcessor.from_pretrained('./pretrained_weights/models--openai--clip-vit-base-patch16', local_files_only=True)
    processor = None
    model = CLIP_DF_Model.init_from_clip('./checkpoints/models--openai--clip-vit-base-patch16')

    # freq_model = model.get_freq_transformer(config['model_config'])
    return model.vision_model, None

def set_additional_default_config(config):
    if not 'frequency_embed' in config['model_config'].keys():
        config['model_config']['frequency_embed'] = 'fft'
    if not 'frequency_patch_num' in config['model_config'].keys():
        config['model_config']['frequency_patch_num'] = 4
    if not 'use_rgb' in config['model_config'].keys():
        config['model_config']['use_rgb'] = True
    if not 'contrast_loss' in config['model_config'].keys():
        config['model_config']['contrast_loss'] = False
    return config