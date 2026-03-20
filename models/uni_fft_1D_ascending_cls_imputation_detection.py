# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

from layers.dcnv4_1D import DCNv3_1D
from layers.fft_seek import PeriodEstimator

class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.__init__params()
    
    def __init__params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
    
    def _get_statistics(self, x):
        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        
    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x
    
    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
    
    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError(f'Not support {mode} mode in RevIN')
        return x

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

class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a1 = nn.Sequential(
            nn.Conv1d(dim // 4, dim // 4, 1),
            nn.GELU(),
            DCNv3_1D(channels=dim//4, kernel_size=7, dilation=1, group=dim // 4)
        )
        self.v1 = nn.Conv1d(dim // 4, dim // 4, 1)
        self.v11 = nn.Conv1d(dim // 4, dim // 4, 1)
        self.v12 = nn.Conv1d(dim // 4, dim // 4, 1)
        self.conv3_1 = nn.Conv1d(dim // 4, dim // 4, 3, padding=1, groups=dim//4)

        self.norm2 = LayerNorm(dim // 2, eps=1e-6, data_format="channels_first")
        self.a2 = nn.Sequential(
            nn.Conv1d(dim // 2, dim // 2, 1),
            nn.GELU(),
            DCNv3_1D(channels=dim//2, kernel_size=9, dilation=1, group=dim // 2)
        )
        self.v2 = nn.Conv1d(dim//2, dim//2, 1)
        self.v21 = nn.Conv1d(dim // 2, dim // 2, 1)
        self.v22 = nn.Conv1d(dim // 4, dim // 4, 1)
        self.proj2 = nn.Conv1d(dim // 2, dim // 4, 1)
        self.conv3_2 = nn.Conv1d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)

        self.norm3 = LayerNorm(dim * 3 // 4, eps=1e-6, data_format="channels_first")
        self.a3 = nn.Sequential(
            nn.Conv1d(dim * 3 // 4, dim * 3 // 4, 1),
            nn.GELU(),
            DCNv3_1D(channels=dim * 3 // 4, kernel_size=11, dilation=1, group=dim * 3 // 4)
        )
        self.v3 = nn.Conv1d(dim * 3 // 4, dim * 3 // 4, 1)
        self.v31 = nn.Conv1d(dim * 3 // 4, dim * 3 // 4, 1)
        self.v32 = nn.Conv1d(dim // 4, dim // 4, 1)
        self.proj3 = nn.Conv1d(dim * 3 // 4, dim // 4, 1)
        self.conv3_3 = nn.Conv1d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)
        self.dim = dim

    def forward(self, x):
        x = self.norm1(x)
        x_split = torch.split(x, self.dim // 4, dim=1)
        a1 = self.a1(x_split[0])
        if a1.shape[1] != self.dim // 4: a1 = a1.permute(0, 2, 1)
        mul1 = a1 * self.v1(x_split[0])
        mul1 = self.v11(mul1)
        x1 = self.conv3_1(self.v12(x_split[1]))
        x1 = x1 + a1
        x1 = torch.cat((x1, mul1), dim=1)

        x1 = self.norm2(x1)
        a2 = self.a2(x1)
        if a2.shape[1] != self.dim // 2: a2 = a2.permute(0, 2, 1)
        mul2 = a2 * self.v2(x1)
        mul2 = self.v21(mul2)
        x2 = self.conv3_2(self.v22(x_split[2]))
        x2 = x2 + self.proj2(a2)
        x2 = torch.cat((x2, mul2), dim=1)

        x2 = self.norm3(x2)
        a3 = self.a3(x2)
        if a3.shape[1] != self.dim * 3 // 4: a3 = a3.permute(0, 2, 1)
        mul3 = a3 * self.v3(x2)
        mul3 = self.v31(mul3)
        x3 = self.conv3_3(self.v32(x_split[3]))
        x3 = x3 + self.proj3(a3)
        x = torch.cat((x3, mul3), dim=1)
        return x

class Block(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., mlp_ratio=4, layer_scale_init_value=1e-5):
        super().__init__()
        self.rfa = ConvMod(dim)
        self.mlp = MLPLayer(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale.unsqueeze(-1) * self.rfa(x))
        x = x.permute(0, 2, 1)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x.permute(0, 2, 1)

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        
        dims = getattr(configs, 'dims', [64, 128, 256, 512])
        depths = getattr(configs, 'depths', [2, 2, 8, 2])
        drop_path_rate = getattr(configs, 'drop_path', 0.)
        layer_scale_init_value = 1e-6
        drop = getattr(configs, 'dropout', 0.)

        self.revin = RevIN(self.enc_in)
        self.fft_estimator = PeriodEstimator(top_k=3)

        self.downsample_layers = nn.ModuleList() 
        stem = nn.Sequential(
            nn.Conv1d(self.enc_in, dims[0] // 2, kernel_size=3, stride=1, padding=1),
            LayerNorm(dims[0] // 2, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv1d(dims[0] // 2, dims[0], kernel_size=3, stride=1, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.Dropout(drop)
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv1d(dims[i], dims[i+1], kernel_size=3, stride=2, padding=1)
            ))

        self.stages = nn.ModuleList() 
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            self.stages.append(nn.Sequential(*[
                Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) 
                for j in range(depths[i])
            ]))
            cur += depths[i]

        self.last_len = self.seq_len
        for _ in range(3): self.last_len = (self.last_len - 1) // 2 + 1
        
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(dims[-1] * self.last_len, self.seq_len * self.enc_in)
        elif self.task_name == 'imputation':
            self.up_layers = nn.ModuleList([
                nn.ConvTranspose1d(dims[i], dims[i], kernel_size=2, stride=2) for i in range(3, 0, -1)
            ])
            self.refine_layers = nn.ModuleList([
                nn.Conv1d(dims[i] + dims[i-1], dims[i-1], kernel_size=1) for i in range(3, 0, -1)
            ])
            self.imputation_head = nn.Conv1d(dims[0], self.enc_in, kernel_size=3, padding=1)
        elif self.task_name == 'classification':
            self.class_head = MLPLayer(dims[-1] * self.last_len, dims[-1], getattr(configs, 'num_class', 10), drop=drop)
        
        self.apply(self._init_weights)
        self.apply(self._init_deform_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def _init_deform_weights(self, m):
        if isinstance(m, DCNv3_1D): m._reset_parameters()

    def update_stage_dilations(self, stage, dilations):
        safe_p = list(dilations) if not isinstance(dilations, list) else dilations.copy()
        if len(safe_p) == 0: safe_p = [1, 1, 1]
        while len(safe_p) < 3: safe_p.append(safe_p[-1])
        for block in stage:
            if hasattr(block, 'rfa'):
                rfa = block.rfa
                rfa.a1[2].dilation = safe_p[0] 
                rfa.a2[2].dilation = safe_p[1]
                rfa.a3[2].dilation = safe_p[2]

    def forward_features(self, x):
        topk_periods = self.fft_estimator(x) 
        current_stride = 1
        multi_scale_features = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            if i > 0: current_stride *= 2
            self.update_stage_dilations(self.stages[i], [max(1, p//current_stride) for p in topk_periods])
            x = self.stages[i](x)
            if self.task_name == 'imputation': multi_scale_features.append(x)
        return multi_scale_features if self.task_name == 'imputation' else x

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'imputation':
            # 修正统计量计算：使用 keepdim=True 保证直接输出 [B, 1, C]
            counts = (torch.sum(mask == 1, dim=1, keepdim=True) + 1e-5)
            means = torch.sum(x_enc, dim=1, keepdim=True) / counts
            x_enc = (x_enc - means).masked_fill(mask == 0, 0)
            stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1, keepdim=True) / counts + 1e-5)
            x = x_enc / stdev
        elif self.task_name != 'classification':
            x = self.revin(x_enc, mode='norm')
        else:
            x = x_enc
            
        x = x.permute(0, 2, 1) 
        features = self.forward_features(x)
        
        if self.task_name == 'imputation':
            x = features[3]
            for i in range(3):
                x = self.up_layers[i](x)
                target_feat = features[2-i]
                if x.shape[-1] != target_feat.shape[-1]:
                    x = F.interpolate(x, size=target_feat.shape[-1], mode='linear', align_corners=False)
                x = torch.cat([x, target_feat], dim=1)
                x = self.refine_layers[i](x)
            
            # D. 映射回原通道 [B, C, L]
            out = self.imputation_head(x)
            # E. 转置回 [B, L, C] -> 形状 [16, 96, 321]
            out = out.permute(0, 2, 1)
            
            # F. 广播还原：stdev 是 [16, 1, 321]，会自动广播到 96 位
            out = out * stdev + means
            return out

        elif self.task_name == 'anomaly_detection':
            x = features.reshape(features.shape[0], -1) 
            x = self.projection(x).reshape(x.shape[0], self.seq_len, self.enc_in)
            return self.revin(x, mode='denorm')

        elif self.task_name == 'classification':
            return self.class_head(features.reshape(features.shape[0], -1))

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None] * x + self.bias[:, None]