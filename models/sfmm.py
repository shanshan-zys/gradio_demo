
import yaml
import torch
from .dfb import ClipMultiDetector
from torchvision import transforms as T
import numpy as np
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F
import matplotlib.pyplot as plt




def init_sfmm_model(device='cuda'):
    
    model_config = './checkpoints/sfmm/clip_freq_multi_contrast.yaml'
    with open(model_config, 'r') as f:
        config = yaml.safe_load(f)

    model = ClipMultiDetector(config).to(device)
    weights_path = './checkpoints/sfmm/ckpt_best.pth'
    ckpt = torch.load(weights_path, map_location=device) 
    model.load_state_dict(ckpt, strict=True)
    
    return model

def preprocess(image):
    def normalize(img):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)

    image = np.array(image)
    image_trans = normalize(T.ToTensor()(image))
    
    return image_trans



def gaussian_attention_map(attention_map, kernel_size=3, sigma=5):
    attention_map = attention_map.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    # smooth_attention_map = F.avg_pool2d(attention_map, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    
    attention_map = F.interpolate(attention_map, [224,224])
    attention_map = gaussian_filter(attention_map[0, 0].cpu().numpy(), sigma=sigma)
    
    # Normalize the blurred attention map
    attention_map = torch.from_numpy(attention_map)
    attention_map /= attention_map.sum()

    smooth_attention_map = attention_map.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions back

    
    return smooth_attention_map.squeeze().numpy()

def smooth_attention_map(attention_map):
    attention_map = attention_map.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    attention_map /= attention_map.sum()
    attention_map *= 16
    print('freq_attn', attention_map)
    attention_map = F.interpolate(attention_map, [224,224])
    
    return attention_map.squeeze()


def analyze_sfmm_attention(input_img, sfmm_output):
    attention = sfmm_output['attentions'][0].squeeze(0).detach().cpu()

    fig, axs = plt.subplots(1, 4, figsize=(10, 3))
    avg_attention = attention.mean(dim=0)

    spatial_attention_map = avg_attention[1][2:198].view(14, 14)
    smooth_spatial_attention_map = gaussian_attention_map(spatial_attention_map)

    freq_to_space_attn_map = avg_attention[200:].mean(dim=0)[2:198].view(14, 14)
    smooth_freq_to_space_attn_map = gaussian_attention_map(freq_to_space_attn_map)


    spa_to_space_attn_map = avg_attention[2:198].mean(dim=0)[2:198].view(14, 14)
    smooth_spa_to_space_attn_map = gaussian_attention_map(spa_to_space_attn_map)

    raw_img = input_img

    axs[0].imshow(raw_img)
    axs[0].set_title('Input Image')
    axs[1].imshow(raw_img, alpha=0.5)
    axs[1].set_title('[CLS] Tokens')
    axs[1].imshow(smooth_spatial_attention_map, cmap='coolwarm', alpha=0.5)
    # axs[2].imshow(fft_img)
    # axs[2].set_title('Frequency')
    # axs[3].imshow(fft_img, alpha=0.5)
    # axs[3].imshow(freq_attention_map, cmap='coolwarm', alpha=0.5)
    axs[2].imshow(raw_img, alpha=0.5)
    axs[2].imshow(smooth_spa_to_space_attn_map, cmap='coolwarm', alpha=0.5)
    axs[2].set_title('Spatial Tokens')

    axs[3].imshow(raw_img, alpha=0.5)
    axs[3].imshow(smooth_freq_to_space_attn_map, cmap='coolwarm', alpha=0.5)
    axs[3].set_title('Freqency Tokens')

    for ax in axs:
        ax.axis('off')

    save_path = f'./outputs/test.png'
    fig.savefig(save_path)
    
    return save_path
