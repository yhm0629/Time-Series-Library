import torch.nn as nn
from .uni_fft_1D_forecast_ascending_order import UniConvNet_A as UniConvNet_A_func


class Model(nn.Module):
    """
    UniConvNet_A model for time series forecasting.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.model = UniConvNet_A_func(
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            enc_in=configs.enc_in,
            drop=getattr(configs, 'dropout', 0.0)
        )
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.model(x_enc)
