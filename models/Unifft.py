import torch.nn as nn
from .uni_fft_1D_forecast_ascending_order_1Dconv import Model as BaseModel

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.model = BaseModel(configs)
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
