import torch
import torch.nn as nn

from models.modules import MLP, ECA

    
class FusionNetwork(nn.Module):
    """
        seqLs: sequence length for each modality. If use pooling feature, each should be 1; otherwise, should be the spatial resolution (h*w) of the feature map.
    """
    def __init__(self, model_config):
        super(FusionNetwork, self).__init__()

        self.projector = MLP(**model_config['projector'])
        
        in_ch= model_config['dim_in']
        k_size = model_config['k_size']
        self.eca = ECA(in_ch, k_size=k_size)

    def forward(self, img_feats, text_feats):
        # img_feats: [B, C1]
        # text_feats: [B, C2]
        features = torch.cat([img_feats, text_feats], dim=1) # [B, C]

        features = self.eca(self.projector(features)) # [B, C]
 
        return features
