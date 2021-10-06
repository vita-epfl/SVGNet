import torch
import torchvision as tv
import typing as th
from .template import RasterModel
from deepsvg.model.model import SVGTransformer


# class ModelTrajectory(torch.nn.Module):
#     def __init__(self,model_cfg,dim_z=128):
#         super().__init__()
#         self.model_cfg = model_cfg
#         self.model_cfg.model_cfg.dim_z = dim_z
#         self.model_cfg.model_cfg.max_num_groups = self.model_cfg.max_num_groups
#         self.model_cfg.model_cfg.max_seq_len = self.model_cfg.max_seq_len
#         self.model = SVGTransformer(self.model_cfg.model_cfg)
#         print(self.model.encoder)

#     def forward(self,x):
#         commands_enc,args_enc, commands_dec, args_dec,params , encode_mode = x
#         return self.model(commands_enc,args_enc, commands_dec, args_dec,params=params , encode_mode=encode_mode).squeeze(0).squeeze(0)
    
    

class ModelTrajectory(RasterModel):
    def __init__(self, model_cfg, data_config, modes=1, future_len=None, in_channels=None, pretrained=True):
        super().__init__(config=None, modes=modes, future_len=future_len, in_channels=in_channels)
        self.model_cfg = model_cfg
        self.model_cfg.model_cfg.dim_z = self.out_dim
        self.model_cfg.model_cfg.max_num_groups = self.model_cfg.max_num_groups
        self.model_cfg.model_cfg.max_seq_len = self.model_cfg.max_seq_len
        self.model = SVGTransformer(self.model_cfg.model_cfg)

    def _forward(self, x):
        commands_enc, args_enc, commands_dec, args_dec, params, encode_mode = x
        return self.model(commands_enc, args_enc, commands_dec, args_dec, params=params,
                          encode_mode=encode_mode).squeeze(0).squeeze(0)