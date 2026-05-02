import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .forgery_clip_encoder import ForgeryCLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')



def build_df_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'pretrained_df_pth', None)
    print(f'---BUILD DF VISION TOWER---- \n vision tower == {vision_tower}')
    # is_absolute_path_exists = os.path.exists(vision_tower)
    return ForgeryCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
