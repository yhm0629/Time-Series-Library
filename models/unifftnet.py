# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

from layers.dcnv_1D import DCNv3_1D
from layers.fft_seek import PeriodEstimator

class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5,affine=True):
        super(RevIN, self).__init__()
        self.num_features =num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.__init__params()
    
    def __init__params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
    
    def _get_statistics(self, x):
        # x: (B, L, C)
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
        
            
class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
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
                 # act_layer='GELU',
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


class ConvMod(nn.Module):
    def __init__(self, dim, num_periods=6): 
        super().__init__()
        self.dim = dim
        self.num_periods = num_periods
        
        # 每一个分支都接收并输出完整的 dim，绝不截断！
        self.a_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, dim, 1),
                nn.GELU(),
                # 感受野依次递增，完美对应从高频到低频的周期
                DCNv3_1D(channels=dim, kernel_size=7 + i * 2, dilation=1, group=dim)
            ) for i in range(num_periods)
        ])
        
        self.v_branches = nn.ModuleList([
            nn.Conv1d(dim, dim, 1) for _ in range(num_periods)
        ])
        
        self.out_projs = nn.ModuleList([
            nn.Conv1d(dim, dim, 1) for _ in range(num_periods)
        ])

    def forward(self, x):
        # x shape: (N, dim, L)
        res = 0 # 累加器
        
        for i in range(self.num_periods):
            # 所有周期分支，均能获取 100% 的全局空间信息
            a = self.a_branches[i](x)
            v = self.v_branches[i](x)
            
            # 幅值与相位的隐式调制 (Implicit Modulation)
            m = a * v 
            out = self.out_projs[i](m)
            
            # 【核心突破】：自适应叠加 (Adaptive Superposition)
            # 代替 concat，避免维度爆炸，且允许波形在隐空间发生相长/相消干涉
            res = res + out 
            
        return res



class Block(nn.Module):
    def __init__(self, dim,
                 drop=0.,
                 drop_path=0.,
                 mlp_ratio=4,
                 layer_scale_init_value=1e-5,
                 num_periods=6,  # 添加 num_periods 参数
                 ):
        super().__init__()

        self.rfa = ConvMod(dim, num_periods=num_periods)  # 传递 num_periods 参数
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
        self.norm1 = build_norm_layer(dim, 'LN')
        self.norm2 = build_norm_layer(dim, 'LN')
        
        # self.dcn = core_op(
        #     channels=dim,
        #     kernel_size=3,
        #     stride=1,
        #     pad=1,
        #     dilation=1,
        #     group=dim // 8,
        #     offset_scale=1.0,
        #     act_layer='GELU',
        #     norm_layer='LN',
        # )

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale.unsqueeze(-1)* self.rfa(x))
        
        # x = x.permute(0, 2, 3, 1)
        # x = x + self.drop_path(self.gamma1 * self.dcn(self.norm1(x)))
        x = x.permute(0, 2, 1)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x.permute(0, 2, 1)


class Unifft(nn.Module):
    r""" UniConvNet
        A PyTorch impl of : `UniConvNet`  -


    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, seq_len=96, pred_len=96, enc_in=3, num_classes=1000, 
                 depths=[2, 2, 8, 2], dims=[64, 128, 256, 512], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1., drop=0., **kwargs
                 ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        # self.adaptive_pool = nn.AdaptiveAvgPool1d(self.last_len)
        self.revin = RevIN(enc_in)
        self.fft_estimator = PeriodEstimator(top_k=6)  # 根据指南使用6个周期

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        # 【关键修改 1】：Stem 层的输入通道必须被强制锁定为 1！
        # 因为我们会在 forward 中把多变量展平为单变量的 batch。
        stem = nn.Sequential(
            nn.Conv1d(1, dims[0] // 2, kernel_size=3, stride=1, padding=1),
            LayerNorm(dims[0] // 2, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv1d(dims[0] // 2, dims[0], kernel_size=3, stride=1, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.Dropout(drop)
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv1d(dims[i], dims[i+1], kernel_size=3, stride=2, padding=1)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        # self.head = nn.Linear(dims[-1], num_classes)
        self.last_len = seq_len // 8
        if self.last_len ==0: self.last_len = 1
        # 【关键修改 2】：Head 的输出维度不再乘以 enc_in，因为现在是单变量进、单变量出
        self.head = nn.Linear(dims[-1] * self.last_len, pred_len)

        self.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)
    
    def update_stage_dilations(self, stage, dilations):
        """
        遍历当前 Stage 的所有 Block，将计算好的 dilation 注入到 ConvMod 的 DCN 中
        dilations: list [d1, d2, d3, ...] 对应 Top-K 周期
        """
        for block in stage:
            # 找到 ConvMod (即 block.rfa)
            if hasattr(block, 'rfa'):
                rfa = block.rfa
                
                # 新的 ConvMod 有多个 a_branches，每个分支对应一个周期
                # 结构是 nn.ModuleList 包含多个 Sequential(Conv, GELU, DCN)
                # DCN 在索引 [2]
                
                # 为每个分支注入对应的 dilation
                for i in range(min(len(rfa.a_branches), len(dilations))):
                    # 获取第 i 个分支的 DCN 层
                    dcn_layer = rfa.a_branches[i][2]  # Sequential 的第三个元素是 DCN
                    # 注入 dilation，注意 dilations 列表的顺序可能需要调整
                    # 根据原始代码，dilations[2] 对应 a1，dilations[1] 对应 a2，dilations[0] 对应 a3
                    # 对于新的结构，我们按顺序分配
                    if i < len(dilations):
                        dcn_layer.dilation = dilations[i]

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def _init_deform_weights(self, m):
        if isinstance(m, DCNv3_1D):
            m._reset_parameters()

    def forward_features(self, x):
        topk_periods = self.fft_estimator(x) # caculate top-k periods
        # print("topk_periods:", topk_periods)
        current_stride=1
        
        for i in range(4):
            x = self.downsample_layers[i](x)
            if i>0:
                current_stride *=2
                
            # calculate dilations for current stage
            stage_dilations=[max(1,p//current_stride) for p in topk_periods]
            
            # inject dilations into DCN layers    
            self.update_stage_dilations(self.stages[i], stage_dilations)
            x = self.stages[i](x)
            
        # return self.norm(x.mean([-1]))  # global average pooling, (N, C, L) -> (N, C)
        return x

    def forward(self, x):
        # x shape: (B, L, C)
        x = self.revin(x, mode='norm')
        B, L, C = x.shape

        # 【核心操作】：通道独立变换 (Channel Independence Transformation)
        # 把 C 个物理变量全部推入 Batch 维度，使其完全解耦
        # (B, L, C) -> (B, C, L) -> (B * C, 1, L)
        x = x.permute(0, 2, 1).reshape(B * C, 1, L)
        
        # 经过所有的特征提取阶段，形状变为 (B*C, dims[-1], L')
        x = self.forward_features(x)
        
        # 展平并预测
        x = x.reshape(B * C, -1)  
        x = self.head(x)  # 输出形状: (B * C, pred_len)
        
        # 【核心操作】：还原物理维度 (Physical Dimension Restoration)
        # (B * C, pred_len) -> (B, C, pred_len) -> (B, pred_len, C)
        x = x.reshape(B, C, self.pred_len).permute(0, 2, 1)
        
        x = self.revin(x, mode='denorm')
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
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


# 将你原来的 class Unifft(nn.Module): 整体替换为以下代码
# 请注意类名必须改为 Model，以便 Time-Series-Library 动态加载

class Model(nn.Module):
    r""" UniConvNet (Adapted for Time-Series-Library)
    支持预测(Forecasting)和异常检测(Anomaly Detection)双任务
    """
    def __init__(self, configs): # 强行接收官方库的 configs
        super().__init__()
        # 1. 从 configs 获取物理量与任务指令
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        # 2. 保留你原有的超参数默认值（如果不从 configs 传，就用原本的默认值）
        depths = getattr(configs, 'depths', [2, 2, 8, 2])
        dims = getattr(configs, 'dims', [64, 128, 256, 512])
        drop_path_rate = getattr(configs, 'drop_path', 0.)
        layer_scale_init_value = 1e-6
        drop = getattr(configs, 'dropout', 0.)

        self.revin = RevIN(self.enc_in)
        self.fft_estimator = PeriodEstimator(top_k=6)

        # ====== 以下特征提取网络保留你原汁原味的逻辑 ======
        self.downsample_layers = nn.ModuleList() 
        stem = nn.Sequential(
            nn.Conv1d(1, dims[0] // 2, kernel_size=3, stride=1, padding=1),
            LayerNorm(dims[0] // 2, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv1d(dims[0] // 2, dims[0], kernel_size=3, stride=1, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.Dropout(drop)
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv1d(dims[i], dims[i+1], kernel_size=3, stride=2, padding=1)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() 
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        
        # 长度计算逻辑保留
        # 严谨计算下采样后的真实时间步长
        self.last_len = self.seq_len
        # 对应你的 3 个 stride=2, padding=1, kernel=3 的下采样层
        for _ in range(3):
            self.last_len = (self.last_len - 1) // 2 + 1
        
        # 3. ======= 核心改造区：双任务分流头 =======
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # 预测任务头 (映射到未来 pred_len)
            self.head = nn.Linear(dims[-1] * self.last_len, self.pred_len * self.enc_in)
        elif self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            # 异常检测任务头 (映射回当前 seq_len 以计算重构误差)
            self.projection = nn.Linear(dims[-1] * self.last_len, self.seq_len * self.enc_in)

        self.apply(self._init_weights)
        self.apply(self._init_deform_weights)
    
    # 你的注入逻辑保持不变
    def update_stage_dilations(self, stage, dilations):
        for block in stage:
            if hasattr(block, 'rfa'):
                rfa = block.rfa
                for i in range(min(len(rfa.a_branches), len(dilations))):
                    dcn_layer = rfa.a_branches[i][2]
                    if i < len(dilations):
                        dcn_layer.dilation = dilations[i]

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def _init_deform_weights(self, m):
        if isinstance(m, DCNv3_1D):
            m._reset_parameters()

    # 你的特征流转逻辑保持不变
    def forward_features(self, x):
        topk_periods = self.fft_estimator(x) 
        current_stride=1
        
        for i in range(4):
            x = self.downsample_layers[i](x)
            if i>0:
                current_stride *=2
            stage_dilations=[max(1,p//current_stride) for p in topk_periods]
            self.update_stage_dilations(self.stages[i], stage_dilations)
            x = self.stages[i](x)
        return x

    # 4. ======= 标准化前向传播入口 =======
    # 必须接收官方库喂入的 5 个张量，即便你只用第一个 x_enc
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # 接入你的 RevIN 平稳化逻辑
        x = self.revin(x_enc, mode='norm')
        B, L, C = x.shape

        # 【核心操作】：通道独立变换 (Channel Independence Transformation)
        # 把 C 个物理变量全部推入 Batch 维度，使其完全解耦
        # (B, L, C) -> (B, C, L) -> (B * C, 1, L)
        x = x.permute(0, 2, 1).reshape(B * C, 1, L)
        
        # 经过你精妙的 FFT + DCN 骨干网络
        x = self.forward_features(x)
        x = x.reshape(B * C, -1)  # flatten
        
        # 任务分流处理
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            x = self.head(x)
            x = x.reshape(B, C, self.pred_len).permute(0, 2, 1)
        elif self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            x = self.projection(x)
            x = x.reshape(B, C, self.seq_len).permute(0, 2, 1)
        
        # 逆平稳化，还原真实量纲（这对于异常检测的 MSE 也是严谨有效的）
        x = self.revin(x, mode='denorm')
        return x
