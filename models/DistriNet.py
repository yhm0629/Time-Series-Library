import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str, target_slice: slice = None) -> torch.Tensor:
        if mode == 'norm':
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x, target_slice)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _get_statistics(self, x: torch.Tensor):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = x.mean(dim=dim2reduce, keepdim=True).detach()
        self.stdev = x.std(dim=dim2reduce, keepdim=True, unbiased=False).detach() + self.eps

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor, target_slice: slice = None) -> torch.Tensor:
        mean = self.mean
        stdev = self.stdev
        weight = self.affine_weight
        bias = self.affine_bias
        if target_slice is not None:
            mean = mean[:, :, target_slice]
            stdev = stdev[:, :, target_slice]
            if self.affine:
                weight = weight[target_slice]
                bias = bias[target_slice]
        if self.affine:
            x = (x - bias) / (weight + self.eps)
        return x * stdev + mean


class HaarWaveletLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, momentum=0.1):
        super(HaarWaveletLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.momentum = momentum
        
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.haar_weight = nn.Parameter(torch.randn(out_features, in_features, grid_size) * 0.1)
        self.haar_shift = nn.Parameter(torch.randn(in_features, grid_size) * 0.1)
        self.haar_scale = nn.Parameter(torch.ones(in_features, grid_size) * 2.0)
        
        # 固定网格，不使用自适应更新（简化版本）
        self.register_buffer('grid', torch.linspace(-1, 1, grid_size + 1).view(1, -1).expand(in_features, -1))
    
    def _haar_wavelet_function(self, x, scale, shift):
        z = (x - shift) / (scale + 1e-8)
        cond_left = (z >= 0) & (z < 0.5)
        cond_right = (z >= 0.5) & (z < 1.0)
        wavelet_value = cond_left.float() - cond_right.float()
        norm_factor = math.sqrt(2.0 / (scale + 1e-8))
        return wavelet_value * norm_factor

    def _compute_haar_basis(self, x):
        # 支持2D [B, I] 或 3D [B, L, I] 输入
        original_2d = False
        if x.dim() == 2:
            original_2d = True
            x = x.unsqueeze(1)  # [B, 1, I]
        
        B, L, I = x.shape
        G = self.grid_size
        
        x_expanded = x.unsqueeze(-1)  # [B, L, I, 1]
        grid_points = self.grid[:, :-1]  # [I, G]
        grid_points = grid_points.unsqueeze(0).unsqueeze(0)  # [1, 1, I, G]
        
        scale = F.softplus(self.haar_scale).unsqueeze(0).unsqueeze(0)  # [1, 1, I, G]
        shift = self.haar_shift.unsqueeze(0).unsqueeze(0)  # [1, 1, I, G]
        
        haar_basis = torch.zeros(B, L, I, G, device=x.device)
        
        for g in range(G):
            current_shift = shift[:, :, :, g] + grid_points[:, :, :, g]
            current_scale = scale[:, :, :, g]
            haar_basis[:, :, :, g] = self._haar_wavelet_function(
                x_expanded.squeeze(-1), current_scale, current_shift
            )
        
        if original_2d:
            haar_basis = haar_basis.squeeze(1)  # [B, I, G]
        
        return haar_basis

    def forward(self, x):
        # 支持2D [B, I] 或 3D [B, L, I] 输入
        original_2d = False
        if x.dim() == 2:
            original_2d = True
            x = x.unsqueeze(1)  # [B, 1, I]
        
        base_output = F.linear(x, self.base_weight)
        haar_basis = self._compute_haar_basis(x)
        B, L, I, G = haar_basis.shape
        
        haar_weight_reshaped = self.haar_weight.permute(1, 2, 0)  # [I, G, O]
        haar_output = torch.einsum('blig,igo->blo', haar_basis, haar_weight_reshaped)
        output = base_output + haar_output
        
        if original_2d:
            output = output.squeeze(1)  # [B, O]
        
        return output


class SpectralHaarKANBlock(nn.Module):
    def __init__(self, channels, grid_size=5, freq_groups=4):
        super(SpectralHaarKANBlock, self).__init__()
        self.channels = channels
        self.freq_groups = freq_groups
        
        self.haar_real = nn.ModuleList([
            HaarWaveletLayer(channels, channels, grid_size=grid_size)
            for _ in range(freq_groups)
        ])
        self.haar_imag = nn.ModuleList([
            HaarWaveletLayer(channels, channels, grid_size=grid_size)
            for _ in range(freq_groups)
        ])
        
        self.group_weights = nn.Parameter(torch.ones(freq_groups))
        
        self.group_norms_real = nn.ModuleList([
            nn.LayerNorm(channels) for _ in range(freq_groups)
        ])
        self.group_norms_imag = nn.ModuleList([
            nn.LayerNorm(channels) for _ in range(freq_groups)
        ])

    def forward(self, x):
        B, L, C = x.shape
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        real, imag = x_fft.real, x_fft.imag
        V = real.shape[1]
        
        out_real, out_imag = torch.zeros_like(real), torch.zeros_like(imag)
        
        for i in range(self.freq_groups):
            start = (i * V) // self.freq_groups
            end = ((i + 1) * V) // self.freq_groups if i < self.freq_groups - 1 else V
            if start == end:
                continue
            
            real_group = real[:, start:end, :]
            imag_group = imag[:, start:end, :]
            
            real_group_norm = self.group_norms_real[i](real_group)
            imag_group_norm = self.group_norms_imag[i](imag_group)
            
            real_out = self.haar_real[i](real_group_norm)
            imag_out = self.haar_imag[i](imag_group_norm)
            
            weight = F.softmax(self.group_weights, dim=0)[i]
            out_real[:, start:end, :] = real_out * weight
            out_imag[:, start:end, :] = imag_out * weight
        
        out_fft = torch.complex(out_real, out_imag)
        return torch.fft.irfft(out_fft, n=L, dim=1, norm='ortho')


class MLPLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.c_out = configs.c_out
        self.revin = RevIN(self.channels)
        
        self.encoder = nn.ModuleList([
            SpectralHaarKANBlock(
                self.channels,
                grid_size=getattr(configs, 'grid_size', 5),
                freq_groups=getattr(configs, 'freq_groups', 4)
            ) for _ in range(configs.e_layers)
        ])
        
        self.encoder_norms = nn.ModuleList([
            nn.LayerNorm(self.channels) for _ in range(configs.e_layers)
        ])
        
        self.target_slice = slice(0, self.c_out)

        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(self.channels, configs.c_out, bias=True)
        elif self.task_name == 'classification':
            self.classification_head = MLPLayer(
                in_features=self.channels,
                hidden_features=getattr(configs, 'd_model', 256),
                out_features=configs.num_class,
                drop=getattr(configs, 'dropout', 0.1)
            )
        else:
            self.projection = nn.Linear(self.channels, configs.c_out, bias=True)

    def _execute_encoder(self, z):
        for i, layer in enumerate(self.encoder):
            z = z + layer(self.encoder_norms[i](z))
        return z

    def forecast(self, x_enc):
        z = self.revin(x_enc, 'norm')
        z = self._execute_encoder(z)
        
        z = z.permute(0, 2, 1)
        z = self.predict_linear(z).permute(0, 2, 1)
        
        z = self._execute_encoder(z)
        dec_out = self.projection(z)
        return self.revin(dec_out, 'denorm', target_slice=self.target_slice)

    def imputation(self, x_enc):
        z = self.revin(x_enc, 'norm')
        z = self._execute_encoder(z)
        dec_out = self.projection(z)
        return self.revin(dec_out, 'denorm', target_slice=self.target_slice)

    def anomaly_detection(self, x_enc):
        z = self.revin(x_enc, 'norm')
        z = self._execute_encoder(z)
        dec_out = self.projection(z)
        return self.revin(dec_out, 'denorm', target_slice=self.target_slice)

    def classification(self, x_enc):
        z = self.revin(x_enc, 'norm')
        z = self._execute_encoder(z)
        z = torch.mean(z, dim=1)
        return self.classification_head(z)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            return self.imputation(x_enc)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        if self.task_name == 'classification':
            return self.classification(x_enc)
        return None