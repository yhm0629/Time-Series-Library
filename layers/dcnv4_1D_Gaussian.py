import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class GaussianRBFInterpolator1D:
    """
    多通道并行化高斯径向基函数插值算子 (Multi-channel Gaussian RBF Interpolator)
    利用 torch.gather 实现极速的跨域特征提取与平滑梯度回传
    """
    @staticmethod
    def interpolate_fast(x, positions, sigma=1.0, window_size=7):
        # x: [B, C, 1, L] (对应 DCN 中的 [N*group, group_channels, 1, L])
        B, C, _, L = x.shape
        x = x.squeeze(2)  # [B, C, L]

        # positions: [B, 1, num_positions, 1]
        num_positions = positions.shape[2]
        positions_flat = positions.view(B, num_positions)

        # 归一化坐标还原至物理连续索引 [B, num_positions]
        original_indices = (positions_flat + 1.0) / 2.0 * (L - 1.0)

        # 1. 确立物理锚点 (Physical Anchors)
        center_indices = torch.round(original_indices).long()
        half_window = window_size // 2
        window_offsets = torch.arange(-half_window, half_window + 1, dtype=torch.long, device=x.device)

        sample_indices_int = center_indices.unsqueeze(-1) + window_offsets.view(1, 1, -1)  # [B, num_positions, window_size]
        sample_indices_int = torch.clamp(sample_indices_int, 0, L - 1)

        # 2. 跨域提取特征 (Cross-domain Feature Extraction) - 多通道极致优化
        # 将索引扩展至多通道维度: [B, C, num_positions, window_size]
        sample_indices_expanded = sample_indices_int.unsqueeze(1).expand(-1, C, -1, -1)
        # 扩展输入特征: [B, C, num_positions, L]
        x_expanded = x.unsqueeze(-2).expand(-1, -1, num_positions, -1)
        # 使用 gather 沿着 L 维度(dim=-1)精确抓取离散特征
        sampled_values = torch.gather(x_expanded, dim=-1, index=sample_indices_expanded) # [B, C, num_positions, window_size]

        # 3. 核心数学桥梁：连续距离与高斯拉力 (Continuous Distance & Gaussian Pull)
        distances = sample_indices_int.float() - original_indices.unsqueeze(-1)  # [B, num_positions, window_size]
        weights = torch.exp(-distances**2 / (2 * sigma**2))
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 4. 加权求和 (Weighted Summation)
        weights = weights.unsqueeze(1)  # [B, 1, num_positions, window_size]
        sampled = torch.sum(weights * sampled_values, dim=-1)  # [B, C, num_positions]

        return sampled.unsqueeze(2)  # 还原至 [B, C, 1, num_positions]

class LayerNorm1d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 2, 1)
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
        nn.init.constant_(self.offset_linear.weight, 0.)
        nn.init.constant_(self.offset_linear.bias, 0.)
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
        mask = mask 

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
        # grid_y = torch.zeros_like(grid_x)
        # grid = torch.cat([grid_x, grid_y], dim=-1)
        
        # 删掉或者注释这段旧代码
        # with torch.backends.cudnn.flags(enabled=False):
        #     sampled = F.grid_sample(
        #          x_in,
        #          grid,
        #          mode='bilinear',
        #          padding_mode='zeros',
        #          align_corners=True
        #     )
        
        # ========== 核心手术：DSP 原生插值替换 ==========
        # 此时 x_in 维度: [N*group, group_channels, 1, L]
        # 此时 grid_x 维度: [N*group, 1, L*kernel_size, 1]
        
        sampled = GaussianRBFInterpolator1D.interpolate_fast(
            x=x_in, 
            positions=grid_x, 
            sigma=1.0,           # 高斯核标准差，可作为超参传入
            window_size=7        # 感受野窗口大小，必须为奇数
        )
        # ===============================================

        sampled = sampled.view(N, self.group, self.group_channels, L, self.kernel_size)
        sampled = sampled.permute(0, 3, 1, 4, 2) 

        mask = mask.view(N, L, self.group, self.kernel_size, 1)
        output = (sampled * mask).sum(dim=3) 
        output = output.reshape(N, L, C) 

        output = self.output_proj(output)
        
        if is_channels_first:
            output = output.permute(0, 2, 1)  
        
        return output
