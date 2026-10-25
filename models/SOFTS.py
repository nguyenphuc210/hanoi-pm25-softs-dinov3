import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted
# [DINOv3-UPDATE] import encoder mới. Bỏ:
#   from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.DINOv3Encoder import DINOv3Encoder


class STAR(nn.Module):
    """STar Aggregate-Redistribute — encoder gốc của SOFTS.

    [DINOv3-UPDATE] Class này KHÔNG còn được gắn vào Model.
    Giữ lại để đối chiếu với bản upstream (Secilia-Cxy/SOFTS).
    Encoder đang dùng là DINOv3Encoder, xem Model.__init__.
    """

    def __init__(self, d_series, d_core):
        super(STAR, self).__init__()
        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, input, *args, **kwargs):
        batch_size, channels, d_series = input.shape

        combined_mean = F.gelu(self.gen1(input))
        combined_mean = self.gen2(combined_mean)

        if self.training:
            ratio = F.softmax(combined_mean, dim=1)
            ratio = ratio.permute(0, 2, 1)
            ratio = ratio.reshape(-1, channels)
            indices = torch.multinomial(ratio, 1)
            indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)
            combined_mean = torch.gather(combined_mean, 1, indices)
            combined_mean = combined_mean.repeat(1, channels, 1)
        else:
            weight = F.softmax(combined_mean, dim=1)
            combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True).repeat(1, channels, 1)

        combined_mean_cat = torch.cat([input, combined_mean], -1)
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat))
        combined_mean_cat = self.gen4(combined_mean_cat)
        output = combined_mean_cat

        return output, None


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # Embedding — không đổi so với SOFTS gốc.
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)
        self.use_norm = configs.use_norm

        # [DINOv3-UPDATE] begin -------------------------------------------------
        # Trước đây:
        #   self.encoder = Encoder([
        #       EncoderLayer(
        #           STAR(configs.d_model, configs.d_core),
        #           configs.d_model, configs.d_ff,
        #           dropout=configs.dropout, activation=configs.activation,
        #       ) for _ in range(configs.e_layers)
        #   ])
        #
        # `e_layers` nay là số block DINOv3 (ViT-S/16 gốc dùng 12).
        # `d_core`, `d_ff`, `activation` không còn đi vào encoder.
        # Bật trọng số Meta: thêm configs.dinov3_pretrained = True
        # (cần `pip install -U transformers` và mạng để tải checkpoint).
        dinov3_layers = getattr(configs, "dinov3_layers", None)
        if dinov3_layers is None:
            dinov3_layers = configs.e_layers
        self.encoder = DINOv3Encoder(
            d_model=configs.d_model,
            depth=dinov3_layers,
            embed_dim=getattr(configs, "dinov3_dim", 384),
            num_heads=getattr(configs, "dinov3_heads", 6),
            mlp_ratio=getattr(configs, "dinov3_mlp_ratio", 4.0),
            n_register_tokens=getattr(configs, "dinov3_registers", 4),
            layerscale_init=getattr(configs, "dinov3_layerscale", 1e-5),
            dropout=configs.dropout,
            pretrained=getattr(configs, "dinov3_pretrained", False),
            pretrained_model_id=getattr(
                configs,
                "dinov3_model_id",
                "facebook/dinov3-vits16-pretrain-lvd1689m",
            ),
        )
        # [DINOv3-UPDATE] end ---------------------------------------------------

        # Decoder — không đổi: mỗi series token chiếu ra pred_len.
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # [DINOv3-UPDATE] cùng chữ ký Encoder gốc: (tokens, attn_mask) -> (tokens, attns).
        # enc_out vẫn là [B, n_token, d_model]; DINOv3 không đổi số token.
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
