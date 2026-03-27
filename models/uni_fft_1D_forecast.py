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
    def __init__(self, dim):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a1 = nn.Sequential(
            nn.Conv1d(dim // 4, dim // 4, 1),
            nn.GELU(),
            
            # nn.Conv2d(dim // 4, dim // 4, 7, padding=3, groups=dim // 4)
            
            DCNv3_1D(
                channels=dim//4,
                kernel_size=7,
                dilation=1, # use the value of FFT
                group=dim // 4,
            )
        )
        self.v1 = nn.Conv1d(dim // 4, dim // 4, 1)
        self.v11 = nn.Conv1d(dim // 4, dim // 4, 1)
        self.v12 = nn.Conv1d(dim // 4, dim // 4, 1)
        self.conv3_1 = nn.Conv1d(dim // 4, dim // 4, 3, padding=1, groups=dim//4)

        self.norm2 = LayerNorm(dim // 2, eps=1e-6, data_format="channels_first")
        self.a2 = nn.Sequential(
            nn.Conv1d(dim // 2, dim // 2, 1),
            nn.GELU(),
            DCNv3_1D(
                channels=dim//2,
                kernel_size=9,
                dilation=1, # use the value of FFT
                group=dim // 2,
            )
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
            DCNv3_1D(
                channels=dim * 3 // 4,
                kernel_size=11,
                dilation=1,  # use the value of FFT
                group=dim * 3 // 4,
            )
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
        
        # 第一层
        a1 = self.a1(x_split[0])
        if a1.shape[1] != self.dim // 4:  
            a1 = a1.permute(0, 2, 1)  # (N, L, C//4) -> (N, C//4, L)
        mul1 = a1 * self.v1(x_split[0])
        mul1 = self.v11(mul1)
        x1 = self.conv3_1(self.v12(x_split[1]))
        x1 = x1 + a1
        x1 = torch.cat((x1, mul1), dim=1)

        # 第二层
        x1 = self.norm2(x1)
        a2 = self.a2(x1)  # 重新计算a
        if a2.shape[1] != self.dim // 2:  
            a2 = a2.permute(0, 2, 1)  # (N, L, C//2) -> (N, C//2, L)
        mul2 = a2 * self.v2(x1)
        mul2 = self.v21(mul2)
        x2 = self.conv3_2(self.v22(x_split[2]))
        x2 = x2 + self.proj2(a2)
        x2 = torch.cat((x2, mul2), dim=1)

        # 第三层
        x2 = self.norm3(x2)
        a3 = self.a3(x2)  # 重新计算a
        if a3.shape[1] != self.dim * 3 // 4:  
            a3 = a3.permute(0, 2, 1)  # (N, L, C*3//4) -> (N, C*3//4, L)
        mul3 = a3 * self.v3(x2)
        mul3 = self.v31(mul3)
        x3 = self.conv3_3(self.v32(x_split[3]))
        x3 = x3 + self.proj3(a3)
        x = torch.cat((x3, mul3), dim=1)

        return x



class Block(nn.Module):
    def __init__(self, dim,
                 drop=0.,
                 drop_path=0.,
                 mlp_ratio=4,
                 layer_scale_init_value=1e-5,
                 ):
        super().__init__()

        self.rfa = ConvMod(dim)
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
        self.fft_estimator = PeriodEstimator(top_k=3)

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv1d(enc_in, dims[0] // 2, kernel_size=3, stride=1, padding=1),
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
        self.head = nn.Linear(dims[-1] * self.last_len, pred_len * enc_in)

        self.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)
    
    def update_stage_dilations(self, stage, dilations):
        """
        遍历当前 Stage 的所有 Block，将计算好的 dilation 注入到 RFA 的 DCN 中
        dilations: list [d1, d2, d3] 对应 Top-3 周期
        """
        for block in stage:
            # 找到 ConvMod (即 block.rfa)
            if hasattr(block, 'rfa'):
                rfa = block.rfa
                
                # RFA 有三层 (a1, a2, a3)，对应三个周期
                # 结构是 Sequential(Conv, GELU, DCN) -> DCN 在索引 [2]
                
                # 注入 Top-1 周期 -> a1
                if len(dilations) >= 1:
                    rfa.a1[2].dilation = dilations[0]
                
                # 注入 Top-2 周期 -> a2
                if len(dilations) >= 2:
                    rfa.a2[2].dilation = dilations[1]
                
                # 注入 Top-3 周期 -> a3
                if len(dilations) >= 3:
                    rfa.a3[2].dilation = dilations[2]

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
        # x = self.forward_features(x)
        # x = self.head(x)
        x = self.revin(x, mode='norm')
        x = x.permute(0, 2, 1)  # (B, L, C) -> (B, C, L)
        x = self.forward_features(x)
        # x = self.adaptive_pool(x)
        x = x.reshape(x.shape[0], -1)  # flatten
        x = self.head(x)
        x = x.reshape(x.shape[0], self.pred_len, self.enc_in)
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


model_urls = {
    "UniConvNet_A_1k": "https://huggingface.co/ai-modelwithcode/UniConvNet/resolve/main/uniconvnet_a_1k_224.pth",
    "UniConvNet_P0_1k": "https://huggingface.co/ai-modelwithcode/UniConvNet/resolve/main/uniconvnet_p0_1k_224_ema.pth",
    "UniConvNet_P1_1k": "https://huggingface.co/ai-modelwithcode/UniConvNet/resolve/main/uniconvnet_p1_1k_224_ema.pth",
    "UniConvNet_P2_1k": "https://huggingface.co/ai-modelwithcode/UniConvNet/resolve/main/uniconvnet_p2_1k_224_ema.pth",
}


@register_model
def UniConvNet_A(pretrained=False, in_22k=False, **kwargs):
    model = Unifft(depths=[2, 3, 9, 2], dims=[24, 48, 96, 192], **kwargs)
    if pretrained:
        url = model_urls['UniConvNet_A_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def Unifft_P0(pretrained=False, in_22k=False, **kwargs):
    model = Unifft(depths=[2, 2, 7, 2], dims=[32, 64, 128, 256], **kwargs)
    if pretrained:
        url = model_urls['UniConvNet_P0_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def Unifft_P1(pretrained=False, in_22k=False, **kwargs):
    model = Unifft(depths=[2, 3, 6, 3], dims=[32, 64, 128, 256], **kwargs)
    if pretrained:
        url = model_urls['UniConvNet_P1_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def Unifft_P2(pretrained=False, in_22k=False, **kwargs):
    model = Unifft(depths=[3, 3, 11, 3], dims=[32, 64, 128, 256], **kwargs)
    if pretrained:
        url = model_urls['UniConvNet_P2_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

