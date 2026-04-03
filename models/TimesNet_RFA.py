import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
# 确保能正确导入外部库
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Time_Series_Library"))
from layers.Embed import DataEmbedding
from layers.RFABlock_v1 import ConvMod
from layers.fft_seek import PeriodEstimator


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # 强制将 K 锁定为 3，以完美适配你 ConvMod 中硬编码的 3 层级联感受野
        self.k = getattr(configs, 'top_k', 3) 
        if self.k < 3:
            self.k = 3
            
        # 极其严格的硬件维度拦截
        if configs.d_model % 4 != 0:
            raise ValueError(f"configs.d_model 必须是 4 的倍数以适配 ConvMod, 当前值为 {configs.d_model}")
            
        # 核心替换 1：特征提取流形完全由 ConvMod 接管
        self.conv_mod = ConvMod(dim=configs.d_model)
        
        # 核心替换 2：频域感知机制完全由你的 PeriodEstimator 接管
        self.period_estimator = PeriodEstimator(top_k=self.k)

    def forward(self, x):
        # x 的原生输入形状: [B, T, C]
        
        # 1. 数据集张量流形正交旋转 (Orthogonal Rotation of Tensor Manifold)
        # 将时序维度与通道维度互换，完美对接你的 1D 算子标准: [B, T, C] -> [B, C, L]
        x_1d = x.permute(0, 2, 1) 
        
        # 2. 绝对纯粹的频域先验提取 (Pure Frequency-Domain Priori Extraction)
        # 使用你的 fft_seek.py 进行物理周期测算
        period_list = self.period_estimator(x_1d)

        # 3. 优雅的内部状态更新 (Elegant Internal State Update)
        # 调用我们之前约定的高内聚 API，将物理规律直接注入黑盒
        self.conv_mod.update_dilations(period_list)

        # 4. 执行 1D 动态形变提取，单次前向传播 (Single Forward Pass)
        res_1d = self.conv_mod(x_1d) # -> [B, C, L]
        
        # 5. 维度复原归位 (Dimensional Restoration)
        # [B, C, L] -> [B, T, C]
        res_1d = res_1d.permute(0, 2, 1) 

        # 6. 闭环残差连接 (Residual Connection)
        res_final = res_1d + x
        return res_final


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        
        # 根据原始TimesNet实现，predict_linear在forecast和anomaly_detection任务中都需要
        if self.task_name in ['long_term_forecast', 'short_term_forecast', 'anomaly_detection']:
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            
        if self.task_name in ['long_term_forecast', 'short_term_forecast', 'imputation', 'anomaly_detection']:
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc.sub(means)
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # 使用predict_linear扩展序列长度
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        if x_mark_enc is not None:
            # x_mark_enc should be a mask with shape (batch_size, seq_len)
            # If it has more dimensions, we need to handle it properly
            if x_mark_enc.dim() == 3:
                # Take the first feature dimension as mask
                mask = x_mark_enc[:, :, 0].unsqueeze(-1)
            else:
                mask = x_mark_enc.unsqueeze(-1)
            output = output * mask
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None