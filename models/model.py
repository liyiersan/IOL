import torch.nn as nn
from models.modules import MLP
from models.img_encoder import ImgEncoder
from models.fusion import FusionNetwork

class PredictiveModel(nn.Module):
    def __init__(self, model_cfg):
        super(PredictiveModel, self).__init__()

        self.img_encoder = ImgEncoder(model_cfg['img_encoder'])  
        self.text_encoder = MLP(**model_cfg['text_encoder'])

        self.fusion = FusionNetwork(model_cfg['fusion'])

        # prediction head
        self.predict_head = MLP(**model_cfg['predict'])

        self.text_predict = MLP(**model_cfg['text_predict'])


    def forward(self, img, text):
        img_feats = self.img_encoder(img)
        text_feats = self.text_encoder(text)

        text_preds = self.text_predict(text_feats)

        fused_features = self.fusion(img_feats, text_feats) # [B, C]
        predicts = self.predict_head(fused_features) # [B, 1]
      
        return predicts, text_preds