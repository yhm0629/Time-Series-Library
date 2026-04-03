import torch
import torch.nn as nn
import sys
import os

# 添加父目录到系统路径，以便导入Standalone_ConvMod
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layers.RFABlock_v1 import ConvMod

class ConvModSelfAttention(nn.Module):
    """
    Wrapper to use ConvMod as a self-attention equivalent in Transformer-like architectures.
    It adapts the input/output dimensions to (Batch, Length, Channel) as expected by Transformer layers,
    while ConvMod operates on (Batch, Channel, Length).
    """
    def __init__(self, d_model):
        super(ConvModSelfAttention, self).__init__()
        # 确保 d_model 能被 4 整除，因为 ConvMod 内部硬编码了 dim // 4
        if d_model % 4 != 0:
            raise ValueError(f"d_model must be divisible by 4 for ConvMod, got {d_model}")
        self.conv_mod = ConvMod(dim=d_model)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        # 输入形状: (B, L, H, E) 其中 B=batch, L=length, H=heads, E=d_model//n_heads
        # 在 Self-Attention 中，queries == keys == values。我们只需要其中一个作为输入。
        
        B, L, H, E = queries.shape
        
        # 将多头合并为通道维度: (B, L, H, E) -> (B, L, H*E)
        x = queries.reshape(B, L, H * E)
        
        # 转换为 ConvMod 需要的格式: (Batch, Channel, Length)
        x = x.permute(0, 2, 1)  # (B, H*E, L)
        
        # 经过 ConvMod 核心算子
        out = self.conv_mod(x)
        
        # 转换回多头格式: (B, H*E, L) -> (B, L, H*E)
        out = out.permute(0, 2, 1)
        
        # 重新分割为多头: (B, L, H*E) -> (B, L, H, E)
        out = out.reshape(B, L, H, E)
        
        # FEDformer 的注意力层期望返回 (Output, Attention_Map)。
        # 由于 ConvMod 不产生显式的注意力图，我们返回 None。
        return out, None