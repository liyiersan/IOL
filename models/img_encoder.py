import torch.nn as nn
import models.backbones as backbones
from models.modules import CLA, MLP, BasicConv2D

class ImgEncoder(nn.Module):
    """
    Image encoder that uses dynamic backbone.
    """
    def __init__(self, model_config):
        super(ImgEncoder, self).__init__()
        self.ealy_conv = BasicConv2D(in_c=16, out_c=3, kernel_size=1) # [B, 16, H, W] -> [B, 3, H, W]  
        model_name = model_config['model_name']
        params = model_config['params']
        self.backbone = getattr(backbones, model_name)(**params)
        
        ch_list = model_config['ch_list']
        k_size_list = model_config['k_size_list']
        # cross layer attention
        self.cla_list = nn.ModuleList([CLA(ch, k_size) for (ch, k_size) in zip(ch_list, k_size_list)])

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.projector = MLP(**model_config['img_projector'])

    def forward(self, imgs):
        b = imgs.size(0)
        imgs = self.ealy_conv(imgs)
        feature_list = self.backbone(imgs) # four hierarchical features
        for i in range(1, len(self.cla_list)+1):
            feature_list[i] = self.cla_list[i-1](feature_list[i-1], feature_list[i])
        pooled_feature = self.avgpool(feature_list[-1]).reshape(b, -1) # [B, C]
        projected_feature = self.projector(pooled_feature) # [B, C]
        return projected_feature        