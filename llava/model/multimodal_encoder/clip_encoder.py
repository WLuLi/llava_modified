import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers import Dinov2Config, Dinov2Model, AutoImageProcessor


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        # vision tower: a name
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        self.alphas = nn.Parameter(torch.ones(24))
        self.beta = nn.Parameter(torch.ones(6))

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
            self.clip_hidden_size = self.cfg_only.hidden_size
            self.dino_v2_cfg_only = Dinov2Config.from_pretrained("facebook/dinov2-base")
            self.dino_v2_hidden_size = self.dino_v2_cfg_only.hidden_size

        self.layer_norms_clip = nn.ModuleList([nn.LayerNorm(self.clip_hidden_size) for _ in range(24)])
        self.linears_clip = nn.ModuleList([nn.Linear(self.clip_hidden_size, self.clip_hidden_size) for _ in range(24)])

        self.layer_norms_dinov2 = nn.ModuleList([nn.LayerNorm(self.dino_v2_hidden_size) for _ in range(6)])
        self.linears_dinov2 = nn.ModuleList([nn.Linear(self.dino_v2_hidden_size, self.dino_v2_hidden_size) for _ in range(6)])

        # align dinov2 and clip
        self.mlp = nn.Sequential(
            nn.Linear(self.dino_v2_hidden_size, self.clip_hidden_size),
            nn.ReLU(),
            nn.Linear(self.clip_hidden_size, self.clip_hidden_size),
        )

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)
        self.clip_hidden_size = self.vision_tower.config.hidden_size

        self.dino_v2_image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.dino_v2 = Dinov2Model.from_pretrained("facebook/dinov2-base")
        self.dino_v2_hidden_size = self.dino_v2.config.hidden_size

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

    def compute_clip_features(self, image_forward_outs):
        features = []
        for i in range(24):
            layer_output = self.feature_select(image_forward_outs.hidden_states[i])
            normed_output = self.layer_norms_clip[i](layer_output)
            linear_output = self.linears_clip[i](normed_output)
            features.append(self.alphas[i] * linear_output)
        return sum(features)

    def compute_dinov2_features(self, image_forward_outs):
        features = []
        for i in range(19, 25):
            layer_output = self.feature_select(image_forward_outs.hidden_states[i])
            normed_output = self.layer_norms_dinov2[i - 19](layer_output)
            linear_output = self.linears_dinov2[i - 19](normed_output)
            features.append(self.betas[i - 19] * linear_output)
        return sum(features)

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                clip_image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                clip_image_feature = self.compute_clip_features(clip_image_forward_out)
                
                image = self.dino_v2_image_processor(image)
                dino_v2_image_forward_out = self.dino_v2(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                dino_v2_image_feature = self.compute_dinov2_features(dino_v2_image_forward_out)
                
                image_features.append(self.mlp(dino_v2_image_feature) + clip_image_feature)
        else:
            clip_image_forward_out = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            clip_image_feature = self.compute_clip_features(clip_image_forward_out)

            dino_v2_image_forward_out = self.dino_v2(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            dino_v2_image_feature = self.compute_dinov2_features(dino_v2_image_forward_out)

            image_features = self.mlp(dino_v2_image_feature) + clip_image_feature

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.clip_hidden_size, device=self.device, dtype=self.dtype)

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
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
