import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

# 请确保这些引用路径与你的项目结构一致
from layers.dcnv_1D import DCNv3_1D
from layers.fft_seek import PeriodEstimator

# ==========================================
# 辅助模块 (RevIN, LayerNorm 等保持不变)
# ==========================================
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
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
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
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

def build_norm_layer(dim, norm_layer, eps=1e-6):
    if norm_layer == 'LN':
        return LayerNorm(dim, eps=eps, data_format="channels_first")
    raise NotImplementedError(f'build_norm_layer does not support {norm_layer}')


class ConvMod_Large(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim


        c1 = dim // 4
        c2 = dim // 2
        c3 = dim * 3 // 4


        g1 = max(1, c1 // 32)
        g2 = max(1, c2 // 32)
        g3 = max(1, c3 // 32)

        if c1 % g1 != 0: g1 = 1
        if c2 % g2 != 0: g2 = 1
        if c3 % g3 != 0: g3 = 1


        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        
        # 分支 1
        self.a1 = nn.Sequential(
            nn.Conv1d(c1, c1, 1),
            nn.GELU(),
            # 【重点】：这里传入计算好的稀疏 group g1
            DCNv3_1D(channels=c1, kernel_size=7, dilation=1, group=g1) 
        )
        self.v1 = nn.Conv1d(c1, c1, 1)
        self.v11 = nn.Conv1d(c1, c1, 1)
        self.v12 = nn.Conv1d(c1, c1, 1)
        self.conv3_1 = nn.Conv1d(c1, c1, 3, padding=1, groups=c1)

        self.norm2 = LayerNorm(dim // 2, eps=1e-6, data_format="channels_first")
        
        # 分支 2
        self.a2 = nn.Sequential(
            nn.Conv1d(c2, c2, 1),
            nn.GELU(),
            # 【重点】：这里传入 g2
            DCNv3_1D(channels=c2, kernel_size=9, dilation=1, group=g2)
        )
        self.v2 = nn.Conv1d(c2, c2, 1)
        self.v21 = nn.Conv1d(c2, c2, 1)
        self.v22 = nn.Conv1d(c1, c1, 1)
        self.proj2 = nn.Conv1d(c2, c1, 1)
        self.conv3_2 = nn.Conv1d(c1, c1, 3, padding=1, groups=c1)

        self.norm3 = LayerNorm(dim * 3 // 4, eps=1e-6, data_format="channels_first")
        
        # 分支 3
        self.a3 = nn.Sequential(
            nn.Conv1d(c3, c3, 1),
            nn.GELU(),
            # 【重点】：这里传入 g3
            DCNv3_1D(channels=c3, kernel_size=11, dilation=1, group=g3)
        )
        self.v3 = nn.Conv1d(c3, c3, 1)
        self.v31 = nn.Conv1d(c3, c3, 1)
        self.v32 = nn.Conv1d(c1, c1, 1)
        self.proj3 = nn.Conv1d(c3, c1, 1)
        self.conv3_3 = nn.Conv1d(c1, c1, 3, padding=1, groups=c1)

    def forward(self, x):
        # Forward 逻辑和原本完全一致
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

# =========================================================================
# 【重点修改 2】BlockLarge
#  专门调用 ConvMod_Large 的 Block
# =========================================================================
class BlockLarge(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., mlp_ratio=4, layer_scale_init_value=1e-5):
        super().__init__()
        # 【重点】：这里实例化的是 ConvMod_Large
        self.rfa = ConvMod_Large(dim)
        
        self.mlp = MLPLayer(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = build_norm_layer(dim, 'LN')
        self.norm2 = build_norm_layer(dim, 'LN')

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale.unsqueeze(-1) * self.rfa(x))
        x = x.permute(0, 2, 1)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x.permute(0, 2, 1)

# =========================================================================
# 主模型结构：Unifft_Large
# =========================================================================
class Unifft_Large(nn.Module):
    def __init__(self, seq_len=96, pred_len=96, enc_in=3, num_classes=1000, 
                 depths=[3, 3, 18, 3], dims=[160, 320, 640, 1280], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1., drop=0., **kwargs
                 ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.revin = RevIN(enc_in)
        self.fft_estimator = PeriodEstimator(top_k=3)

        self.downsample_layers = nn.ModuleList()
        # Stem
        stem = nn.Sequential(
            nn.Conv1d(enc_in, dims[0] // 2, kernel_size=3, stride=1, padding=1),
            LayerNorm(dims[0] // 2, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv1d(dims[0] // 2, dims[0], kernel_size=3, stride=1, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.Dropout(drop)
        )
        self.downsample_layers.append(stem)
        
        # Downsample layers
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
                # 【重点】：这里使用的是 BlockLarge
                *[BlockLarge(dim=dims[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) 
        
        # Head logic
        self.last_len = seq_len // 8
        if self.last_len == 0: self.last_len = 1
        self.head = nn.Linear(dims[-1] * self.last_len, pred_len * enc_in)

        self.apply(self._init_weights)
        self.apply(self._init_deform_weights)
    
    def update_stage_dilations(self, stage, dilations):
        for block in stage:
            if hasattr(block, 'rfa'):
                rfa = block.rfa
                if len(dilations) >= 1: rfa.a1[2].dilation = dilations[2]
                if len(dilations) >= 2: rfa.a2[2].dilation = dilations[1]
                if len(dilations) >= 3: rfa.a3[2].dilation = dilations[0]

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def _init_deform_weights(self, m):
        if isinstance(m, DCNv3_1D):
            m._reset_parameters()

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

    def forward(self, x):
        x = self.revin(x, mode='norm')
        x = x.permute(0, 2, 1)
        x = self.forward_features(x)
        x = x.reshape(x.shape[0], -1)
        x = self.head(x)
        x = x.reshape(x.shape[0], self.pred_len, self.enc_in)
        x = self.revin(x, mode='denorm')
        return x



@register_model
def Unifft_B(pretrained=False, in_22k=False, **kwargs):
    # [cite_start]Base: 对标 Swin-B [cite: 1408]
    model = Unifft_Large(depths=[4, 4, 13, 4], dims=[112, 224, 448, 896], **kwargs)
    return model

@register_model
def Unifft_L(pretrained=False, in_22k=False, **kwargs):
    # [cite_start]Large: 对标 Swin-L [cite: 1122]
    model = Unifft_Large(depths=[3, 3, 18, 3], dims=[160, 320, 640, 1280], **kwargs)
    return model

@register_model
def Unifft_XL(pretrained=False, in_22k=False, **kwargs):
    # [cite_start]XL: 对标 ConvNeXt-XL [cite: 1489]
    model = Unifft_Large(depths=[3, 3, 22, 3], dims=[160, 320, 640, 1280], **kwargs)
    return model