import timm
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, model_name, pretrained=False, pretrained_path=None):
        super(ResNet, self).__init__()
        cfg = timm.create_model(model_name, features_only=True, pretrained=False).default_cfg
        if pretrained_path is not None:
            cfg['file'] = pretrained_path
        self.model = timm.create_model(model_name, features_only=True, pretrained=pretrained, pretrained_cfg=cfg)
            

    def forward(self, x):
        x = self.model(x)
        return x