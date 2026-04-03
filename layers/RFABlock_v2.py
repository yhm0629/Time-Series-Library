import torch
import torch.nn as nn
import torch.nn.functional as F

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


class DCNv3_1D(nn.Module):
    def __init__(
        self,
        channels=64,
        kernel_size=3,
        stride=1,
        pad=1,
        dilation=1,
        group=4,
        offset_scale=1.0,
        act_layer='GELU',
        norm_layer='LN',
        center_feature_scale=False,
        remove_center=False,
    ):
        super().__init__()
        if channels % group != 0:
            raise ValueError(f'channels must be divisible by group, but got {channels} and {group}')
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.remove_center = int(remove_center)
        self.feature_reduce = nn.Linear(channels, channels // 2)                 
        self.offset_linear = nn.Linear(channels // 2, group * kernel_size * 1) 
        self.mask_linear = nn.Linear(channels // 2, group * kernel_size)
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self.dw_conv = nn.Conv1d(channels, channels, kernel_size,
                                  stride=1, padding=(kernel_size - 1) // 2,
                                  dilation=dilation, groups=channels)
        self._reset_parameters() 

    def _reset_parameters(self):
        # offset
        nn.init.normal_(self.offset_linear.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.offset_linear.bias, 0.)
        # mask
        nn.init.constant_(self.mask_linear.weight, 0.)
        nn.init.constant_(self.mask_linear.bias, 0.)
        nn.init.xavier_uniform_(self.input_proj.weight) 
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(self, x):
        is_channels_first = False
        if x.ndim == 3 and x.shape[1] == self.channels:
            x = x.permute(0, 2, 1) 
            is_channels_first = True
        N, L, C = x.shape 
        
        x_proj = self.input_proj(x) 
        x_feat = x.permute(0, 2, 1) 
        dynamic_padding = ((self.kernel_size - 1) // 2) * self.dilation
        
        x_feat = F.conv1d(
            input=x_feat,
            weight=self.dw_conv.weight,
            bias=self.dw_conv.bias,
            stride=self.stride,
            padding=dynamic_padding,
            dilation=self.dilation,
            groups=self.channels
        )
        x_feat = x_feat.permute(0, 2, 1) 
        
        x_feat_reduced = self.feature_reduce(x_feat)
        offset = self.offset_linear(x_feat_reduced) 
        mask = self.mask_linear(x_feat_reduced).reshape(N, L, self.group, -1) 
        
        ref_p = torch.arange(L, dtype=torch.float32, device=x.device).view(1, L)
        dil_grid = torch.linspace(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2, self.kernel_size, device=x.device) * self.dilation
        base_grid = ref_p.view(1, L, 1, 1) + dil_grid.view(1, 1, 1, -1) 
        
        offset = offset.view(N, L, self.group, self.kernel_size) 
        sampling_locations = base_grid + offset * self.offset_scale
        sampling_locations = torch.remainder(sampling_locations, L)
        sampling_locations_norm = 2.0 * sampling_locations / (L - 1) - 1.0
        
        x_in = x_proj.view(N, L, self.group, self.group_channels)
        x_in = x_in.permute(0, 2, 3, 1).reshape(N * self.group, self.group_channels, 1, L)
        
        sampling_grid = sampling_locations_norm.permute(0, 2, 1, 3) 
        sampling_grid = sampling_grid.reshape(N * self.group, L * self.kernel_size)                  
        
        grid_x = sampling_grid.view(N * self.group, 1, -1, 1)
        grid_y = torch.zeros_like(grid_x)
        grid = torch.cat([grid_x, grid_y], dim=-1) 
        
        with torch.backends.cudnn.flags(enabled=False):
            sampled = F.grid_sample(
                 x_in, 
                 grid, 
                 mode='bilinear',
                 padding_mode='zeros',
                 align_corners=True
            )
            
        sampled = sampled.view(N, self.group, self.group_channels, L, self.kernel_size)
        sampled = sampled.permute(0, 3, 1, 4, 2) 
        mask = mask.view(N, L, self.group, self.kernel_size, 1)
        
        output = (sampled * mask).sum(dim=3) 
        output = output.reshape(N, L, C) 
        output = self.output_proj(output)                 
        
        if is_channels_first:
            output = output.permute(0, 2, 1)                   
        return output


class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a1 = nn.Sequential(
            nn.Conv1d(dim // 4, dim // 4, 1),
            nn.GELU(),
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

    def update_dilations(self, periods):
        """
        更新ConvMod内部所有DCNv3_1D模块的dilation参数
        periods: 包含3个周期值的列表，对应a1, a2, a3的dilation
        """
        if not periods:
            periods = [1, 1, 1]
        
        # 确保有3个值
        safe_periods = periods.copy()
        while len(safe_periods) < 3:
            safe_periods.append(safe_periods[-1] if safe_periods else 1)
        
        # 更新a1中的DCNv3_1D模块
        dcn1 = self.a1[2]  # a1是nn.Sequential: Conv1d -> GELU -> DCNv3_1D
        dcn1.dilation = safe_periods[0]
        dcn1.dw_conv.dilation = safe_periods[0]
        
        # 更新a2中的DCNv3_1D模块
        dcn2 = self.a2[2]
        dcn2.dilation = safe_periods[1]
        dcn2.dw_conv.dilation = safe_periods[1]
        
        # 更新a3中的DCNv3_1D模块
        dcn3 = self.a3[2]
        dcn3.dilation = safe_periods[2]
        dcn3.dw_conv.dilation = safe_periods[2]

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