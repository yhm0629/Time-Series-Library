import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 确保能正确导入外部库
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Time_Series_Library"))

from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding
from layers.fft_seek import PeriodEstimator
from layers.RFABlock_v1 import ConvMod

# =========================================================================
# 规则 1：所有 PyTorch Module 必须定义在全局作用域，以确保模型可被保存和序列化
# =========================================================================

class ConvModSelfAttention(nn.Module):
    def __init__(self, d_model):
        super(ConvModSelfAttention, self).__init__()
        if d_model % 4 != 0:
            raise ValueError(f"d_model must be divisible by 4 for ConvMod, got {d_model}")
        self.conv_mod = ConvMod(dim=d_model)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        # 放弃在参数中传递 dilation，由外部的 _inject_physical_periods 统一接管物理状态
        x = queries.permute(0, 2, 1)  # (B, L, D) -> (B, D, L)
        out = self.conv_mod(x)
        out = out.permute(0, 2, 1)    # (B, D, L) -> (B, L, D)
        return out, None

class DualStreamEncoderLayer(nn.Module):
    def __init__(self, attention, convmod_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(DualStreamEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.global_attention = attention
        self.local_attention = convmod_attention
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        global_out, global_attn = self.global_attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        local_out, local_attn = self.local_attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        
        weights = torch.softmax(torch.stack([self.alpha, self.beta]), dim=0)
        combined_out = weights[0] * global_out + weights[1] * local_out
        
        x = x + self.dropout(combined_out)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), (global_attn, local_attn)

class ConvModConvLayer(nn.Module):
    """受到 ConvMod 保护的特征蒸馏层"""
    def __init__(self, c_in):
        super(ConvModConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in, out_channels=c_in, kernel_size=3, padding=2, padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        
        if c_in % 4 != 0:
            self.proj_in = nn.Conv1d(c_in, (c_in // 4) * 4, kernel_size=1)
            self.proj_out = nn.Conv1d((c_in // 4) * 4, c_in, kernel_size=1)
            self.conv_mod = ConvMod(dim=(c_in // 4) * 4)
            self.use_proj = True
        else:
            self.proj_in = None
            self.proj_out = None
            self.conv_mod = ConvMod(dim=c_in)
            self.use_proj = False
            
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        
        if self.use_proj:
            x_proj = self.proj_in(x)
            x_convmod = self.conv_mod(x_proj)
            x = x + self.proj_out(x_convmod)
        else:
            x_convmod = self.conv_mod(x)
            x = x + x_convmod
            
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.d_model = configs.d_model
        self.seq_len = configs.seq_len
        
        self.period_estimator = PeriodEstimator(top_k=3)

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # 完美使用原生的 Encoder，省去重写的灾难
        self.encoder = Encoder(
            [
                DualStreamEncoderLayer(
                    AttentionLayer(ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False), configs.d_model, configs.n_heads),
                    ConvModSelfAttention(configs.d_model),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvModConvLayer(configs.d_model) for l in range(configs.e_layers - 1)
            ] if configs.distil and ('forecast' in configs.task_name) else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False), configs.d_model, configs.n_heads),
                    AttentionLayer(ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        
        if self.task_name in ['imputation', 'anomaly_detection']:
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def _inject_physical_periods(self, x_enc):
        """
        全局统筹：提取物理周期并直接硬性修改网络拓扑中的膨胀系数 (Hard injection)
        """
        x_for_fft = x_enc.permute(0, 2, 1)
        periods = self.period_estimator(x_for_fft)
        
        safe_p = periods.copy() if periods else [1, 1, 1]
        while len(safe_p) < 3:
            safe_p.append(safe_p[-1])
            
        # 遍历整个网络，无论 ConvMod 藏在 Attention 还是 Distillation 层中，统统拦截并更新
        for name, module in self.named_modules():
            if module.__class__.__name__ == 'ConvMod':
                module.a1[2].dilation = safe_p[0]
                module.a2[2].dilation = safe_p[1]
                module.a3[2].dilation = safe_p[2]

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out

    def short_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        dec_out = dec_out * std_enc + mean_enc
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        if x_mark_enc is not None:
            # x_mark_enc should be a mask with shape (batch_size, seq_len)
            # If it has more dimensions, we need to handle it properly
            if x_mark_enc.dim() == 3:
                # Take the first feature dimension as mask
                mask = x_mark_enc[:, :, 0].unsqueeze(-1)
            else:
                mask = x_mark_enc.unsqueeze(-1)
            output = output * mask  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 第一步：强制计算物理周期并进行全局参数注入
        self._inject_physical_periods(x_enc)

        # 第二步：任务分发，逻辑异常干净
        if self.task_name == 'long_term_forecast':
            dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'short_term_forecast':
            dec_out = self.short_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None