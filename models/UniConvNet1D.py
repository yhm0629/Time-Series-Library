# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# UniConvNet1D: 1D Temporal Version of UniConvNet
# Adapted from UniConvNet for temporal data processing

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

# Note: We don't import ops_dcnv3 since we're using a placeholder for DCNv3_1D

class RevIN(nn.Module):
    """
    Reversible Instance Normalization.
    关键组件：用于消除时间序列的非平稳性。
    """
    def __init__(self, num_features: int, eps=1e-5, affine=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.affine_weight)
        x = x * self.stdev
        x = x + self.mean
        return x

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

class to_channels_first_1d(nn.Module):
    """Convert from channels_last to channels_first for 1D data"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Input: (N, T, C) -> Output: (N, C, T)
        return x.permute(0, 2, 1)


class to_channels_last_1d(nn.Module):
    """Convert from channels_first to channels_last for 1D data"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Input: (N, C, T) -> Output: (N, T, C)
        return x.permute(0, 2, 1)


def build_norm_layer_1d(dim,
                       norm_layer,
                       in_format='channels_last',
                       out_format='channels_last',
                       eps=1e-6):
    """Build normalization layer for 1D data"""
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first_1d())
        layers.append(nn.BatchNorm1d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last_1d())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last_1d())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first_1d())
    else:
        raise NotImplementedError(
            f'build_norm_layer_1d does not support {norm_layer}')
    return nn.Sequential(*layers)


class MLPLayer(nn.Module):
    r""" MLP layer of InternImage
    Args:
        in_features (int): number of input features
        hidden_features (int): number of hidden features
        out_features (int): number of output features
        act_layer (str): activation layer
        drop (float): dropout rate
    """
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 drop=0.):
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


class LayerNorm1D(nn.Module):
    r""" LayerNorm that supports two data formats for 1D data: channels_last or channels_first.
    channels_last corresponds to inputs with shape (batch_size, length, channels)
    channels_first corresponds to inputs with shape (batch_size, channels, length)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x


class ConvMod1D(nn.Module):
    """1D version of ConvMod for temporal data"""
    def __init__(self, dim):
        super().__init__()

        self.norm1 = LayerNorm1D(dim, eps=1e-6, data_format="channels_first")
        # First layer: kernel size 7
        self.a1 = nn.Sequential(
            nn.Conv1d(dim // 4, dim // 4, 1),
            nn.GELU(),
            nn.Conv1d(dim // 4, dim // 4, 7, padding=3, groups=dim // 4)
        )
        self.v1 = nn.Conv1d(dim // 4, dim // 4, 1)
        self.v11 = nn.Conv1d(dim // 4, dim // 4, 1)
        self.v12 = nn.Conv1d(dim // 4, dim // 4, 1)
        self.conv3_1 = nn.Conv1d(dim // 4, dim // 4, 3, padding=1, groups=dim//4)

        self.norm2 = LayerNorm1D(dim // 2, eps=1e-6, data_format="channels_first")
        # Second layer: kernel size 9
        self.a2 = nn.Sequential(
            nn.Conv1d(dim // 2, dim // 2, 1),
            nn.GELU(),
            nn.Conv1d(dim // 2, dim // 2, 9, padding=4, groups=dim // 2)
        )
        self.v2 = nn.Conv1d(dim//2, dim//2, 1)
        self.v21 = nn.Conv1d(dim // 2, dim // 2, 1)
        self.v22 = nn.Conv1d(dim // 4, dim // 4, 1)
        self.proj2 = nn.Conv1d(dim // 2, dim // 4, 1)
        self.conv3_2 = nn.Conv1d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)

        self.norm3 = LayerNorm1D(dim * 3 // 4, eps=1e-6, data_format="channels_first")
        # Third layer: kernel size 11
        self.a3 = nn.Sequential(
            nn.Conv1d(dim * 3 // 4, dim * 3 // 4, 1),
            nn.GELU(),
            nn.Conv1d(dim * 3 // 4, dim * 3 // 4, 11, padding=5, groups=dim * 3 // 4)
        )
        self.v3 = nn.Conv1d(dim * 3 // 4, dim * 3 // 4, 1)
        self.v31 = nn.Conv1d(dim * 3 // 4, dim * 3 // 4, 1)
        self.v32 = nn.Conv1d(dim // 4, dim // 4, 1)
        self.proj3 = nn.Conv1d(dim * 3 // 4, dim // 4, 1)
        self.conv3_3 = nn.Conv1d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)

        self.dim = dim

    def forward(self, x):
        # x shape: (B, C, T)
        x = self.norm1(x)
        x_split = torch.split(x, self.dim // 4, dim=1)
        
        # Layer 1
        a = self.a1(x_split[0])
        mul = a * self.v1(x_split[0])
        mul = self.v11(mul)
        x1 = self.conv3_1(self.v12(x_split[1]))
        x1 = x1 + a
        x1 = torch.cat((x1, mul), dim=1)

        # Layer 2
        x1 = self.norm2(x1)
        a = self.a2(x1)
        mul = a * self.v2(x1)
        mul = self.v21(mul)
        x2 = self.conv3_2(self.v22(x_split[2]))
        x2 = x2 + self.proj2(a)
        x2 = torch.cat((x2, mul), dim=1)

        # Layer 3
        x2 = self.norm3(x2)
        a = self.a3(x2)
        mul = a * self.v3(x2)
        mul = self.v31(mul)
        x3 = self.conv3_3(self.v32(x_split[3]))
        x3 = x3 + self.proj3(a)
        x = torch.cat((x3, mul), dim=1)

        return x


# Note: DCNv3_1D will be implemented separately as it requires significant changes
# For now, we'll use a placeholder or a standard 1D convolution

class DCNv3_1D_Placeholder(nn.Module):
    """Placeholder for DCNv3 1D - using standard 1D convolution for now"""
    def __init__(self, channels=64, kernel_size=3, stride=1, pad=1, 
                 dilation=1, group=4, offset_scale=1.0, 
                 act_layer='GELU', norm_layer='LN'):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv1d(
            channels, channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=pad,
            groups=channels
        )
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()
        
    def forward(self, x):
        # x shape: (N, T, C) - channels_last format
        x = x.permute(0, 2, 1)  # to (N, C, T) for conv
        x = self.conv(x)  # (N, C, T)
        x = x.permute(0, 2, 1)  # back to (N, T, C) for norm
        x = self.norm(x)  # LayerNorm works on last dimension
        x = self.act(x)
        return x  # (N, T, C)


class Block1D(nn.Module):
    """1D version of Block for temporal data"""
    def __init__(self, dim,
                 drop=0.,
                 drop_path=0.,
                 mlp_ratio=4,
                 layer_scale_init_value=1e-5,
                 use_dcnv3=True):
        super().__init__()

        self.attn = ConvMod1D(dim)
        self.mlp = MLPLayer(in_features=dim,
                           hidden_features=int(dim * mlp_ratio),
                           drop=drop)
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True)
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True)
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                       requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Use DCNv3_1D placeholder for now
        self.dcn = DCNv3_1D_Placeholder(
            channels=dim,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=dim // 8,
            offset_scale=1.0,
            act_layer='GELU',
            norm_layer='LN'
        )

    def forward(self, x):
        # x shape: (B, C, T)
        x = x + self.drop_path(self.layer_scale.unsqueeze(-1) * self.attn(x))
        x = x.permute(0, 2, 1)  # to (B, T, C) for DCN
        x = x + self.drop_path(self.gamma1 * self.dcn(x))
        x = x + self.drop_path(self.gamma2 * self.mlp(x))
        return x.permute(0, 2, 1)  # back to (B, C, T)


class UniConvNet1D_Forecasting(nn.Module):
    r""" 
    UniConvNet1D for Time Series Forecasting.
    
    Changes from Classification:
    1. Removed Global Average Pooling to preserve temporal information.
    2. Replaced Classification Head with a Temporal Projection Head.
    3. Fixed input resolution dependency due to Flatten layer.
    4. Added Reversible Instance Normalization (RevIN) for non-stationary time series.
    
    Args:
        in_chans (int): Number of input variates/channels. Default: 7
        seq_len (int): Input look-back window length. Default: 96
        pred_len (int): Prediction horizon length. Default: 96
        depths (tuple(int)): Number of blocks at each stage.
        dims (int): Feature dimension at each stage.
        drop_path_rate (float): Stochastic depth rate.
        layer_scale_init_value (float): Init value for Layer Scale.
        drop (float): Dropout rate.
        use_revin (bool): Whether to use Reversible Instance Normalization. Default: True
        revin_affine (bool): Whether to use affine parameters in RevIN. Default: True
    """
    def __init__(self, in_chans=7, seq_len=96, pred_len=96, 
                 depths=[2, 2, 8, 2], dims=[64, 128, 256, 512], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, drop=0.,
                 use_revin=True, revin_affine=True):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.in_chans = in_chans
        self.use_revin = use_revin
        
        # 1. Reversible Instance Normalization (RevIN)
        if self.use_revin:
            self.revin = RevIN(num_features=in_chans, eps=1e-5, affine=revin_affine)
        
        # 2. Downsampling layers (Stem + 3 stages)
        # Stem has 2 downsampling steps (stride=2 each), then 3 more stages with stride=2
        # Total downsample factor = 2^5 = 32
        self.downsample_factor = 32 
        
        self.downsample_layers = nn.ModuleList()
        
        # Stem
        stem = nn.Sequential(
            nn.Conv1d(in_chans, dims[0] // 2, kernel_size=3, stride=2, padding=1),
            LayerNorm1D(dims[0] // 2, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv1d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
            LayerNorm1D(dims[0], eps=1e-6, data_format="channels_first"),
            nn.Dropout(drop)
        )
        self.downsample_layers.append(stem)
        
        # Intermediate downsampling
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm1D(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv1d(dims[i], dims[i+1], kernel_size=3, stride=2, padding=1)
            )
            self.downsample_layers.append(downsample_layer)

        # 3. Feature Extraction Stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block1D(dim=dims[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)

        # 4. Validation of sequence length
        # Strictly ensure seq_len is divisible by downsample_factor to avoid dimension mismatch
        if seq_len % self.downsample_factor != 0:
            raise ValueError(f"seq_len ({seq_len}) must be divisible by total stride ({self.downsample_factor})")

        # 5. Projection Head (Forecasting Head)
        # We flatten the latent representation (B, C_last, T_last) -> (B, C_last * T_last)
        # Then project to (B, pred_len * in_chans)
        # Finally reshape to (B, pred_len, in_chans)
        
        self.last_time_dim = seq_len // self.downsample_factor
        self.flatten_dim = dims[-1] * self.last_time_dim
        
        # Using a Linear Projection is effective for LTSF (Long-term Time Series Forecasting)
        self.head = nn.Linear(self.flatten_dim, pred_len * in_chans)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        # Input x: (B, C, T) - Conv1d expects channels_first format
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        
        # x is now (B, dims[-1], seq_len // 16)
        # We DO NOT mean pool here. We keep the temporal structure.
        x = x.permute(0, 2, 1) # (B, C, T) -> (B, T, C) for LayerNorm
        x = self.norm(x)
        # Debug: print shape
        # print(f"Debug forward_features output shape: {x.shape}")
        return x

    def forward(self, x):
        # x shape: (B, seq_len, in_chans)
        
        # Step 1: Normalize with RevIN if enabled
        if self.use_revin:
            x = self.revin(x, 'norm')

        # Step 2: Convert to channels_first format for Conv1d
        # (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1)
        
        # Step 3: Backbone processing
        B, C, T = x.shape
        x = self.forward_features(x) # Output: (B, T_downsampled, C_last)
        
        # Step 4: Flatten: (B, T_downsampled * C_last)
        x = x.reshape(B, -1)
        
        # Step 5: Project: (B, pred_len * in_chans)
        x = self.head(x)
        
        # Step 6: Reshape to output format: (B, pred_len, in_chans)
        x = x.reshape(B, self.pred_len, self.in_chans)

        # Step 7: Denormalize with RevIN if enabled
        if self.use_revin:
            x = self.revin(x, 'denorm')
        
        return x


# Model registration functions
@register_model
def UniConvNet1D_A(pretrained=False, **kwargs):
    model = UniConvNet1D_Forecasting(depths=[2, 3, 9, 2], dims=[24, 48, 96, 192], **kwargs)
    return model


@register_model
def UniConvNet1D_P0(pretrained=False, **kwargs):
    model = UniConvNet1D_Forecasting(depths=[2, 2, 7, 2], dims=[32, 64, 128, 256], **kwargs)
    return model


@register_model
def UniConvNet1D_P1(pretrained=False, **kwargs):
    model = UniConvNet1D_Forecasting(depths=[2, 3, 6, 3], dims=[32, 64, 128, 256], **kwargs)
    return model


@register_model
def UniConvNet1D_P2(pretrained=False, **kwargs):
    model = UniConvNet1D_Forecasting(depths=[3, 3, 11, 3], dims=[32, 64, 128, 256], **kwargs)
    return model


@register_model
def UniConvNet1D_N0(pretrained=False, **kwargs):
    model = UniConvNet1D_Forecasting(depths=[2, 2, 7, 2], dims=[48, 96, 192, 384], **kwargs)
    return model


@register_model
def UniConvNet1D_N1(pretrained=False, **kwargs):
    model = UniConvNet1D_Forecasting(depths=[2, 2, 8, 3], dims=[48, 96, 192, 384], **kwargs)
    return model


@register_model
def UniConvNet1D_N2(pretrained=False, **kwargs):
    model = UniConvNet1D_Forecasting(depths=[3, 3, 11, 3], dims=[48, 96, 192, 384], **kwargs)
    return model


@register_model
def UniConvNet1D_N3(pretrained=False, **kwargs):
    model = UniConvNet1D_Forecasting(depths=[3, 3, 19, 3], dims=[48, 96, 192, 384], **kwargs)
    return model


@register_model
def UniConvNet1D_T(pretrained=False, **kwargs):
    model = UniConvNet1D_Forecasting(depths=[3, 3, 15, 3], dims=[64, 128, 256, 512], **kwargs)
    return model


@register_model
def UniConvNet1D_S(pretrained=False, **kwargs):
    model = UniConvNet1D_Forecasting(depths=[3, 3, 17, 3], dims=[80, 160, 320, 640], **kwargs)
    return model


@register_model
def UniConvNet1D_B(pretrained=False, **kwargs):
    model = UniConvNet1D_Forecasting(depths=[4, 4, 13, 4], dims=[112, 224, 448, 896], **kwargs)
    return model


@register_model
def UniConvNet1D_L(pretrained=False, **kwargs):
    model = UniConvNet1D_Forecasting(depths=[3, 3, 18, 3], dims=[160, 320, 640, 1280], **kwargs)
    return model


@register_model
def UniConvNet1D_XL(pretrained=False, **kwargs):
    model = UniConvNet1D_Forecasting(depths=[3, 3, 22, 3], dims=[160, 320, 640, 1280], **kwargs)
    return model

class Model(nn.Module):
    """
    UniConvNet1D model for time series forecasting.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        # 从 configs 中提取参数
        self.model = UniConvNet1D_Forecasting(
            in_chans=configs.enc_in,
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            # 可以添加更多参数，如 use_revin 等
            use_revin=True,  # 默认使用 RevIN
            revin_affine=True
        )
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 只使用 x_enc 参数，忽略其他参数以保持兼容性
        return self.model(x_enc)
