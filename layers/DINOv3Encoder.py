"""DINOv3 encoder used as a drop-in replacement for the SOFTS Encoder.

Architecture follows the official DINOv3 ViT-S/16 backbone
(facebookresearch/dinov3, ``dinov3_vits16``):

- embed dim 384, 6 heads, MLP ratio 4, GELU (not SwiGLU; that is ViT-S+)
- pre-norm residual block + LayerScale (init 1e-5)
- 1 CLS token + 4 register / storage tokens
- 2D axial RoPE (base 100, coords normalized per axis into [-1, 1])
  applied only to patch tokens; prefix tokens are not rotated
- final LayerNorm

SOFTS feeds a sequence of series tokens ``[B, L, d_model]``, not an image.
Those tokens are linearly adapted into the DINOv3 width, packed onto a
near-square grid (so the 2D RoPE has a spatial layout), run through the
DINOv3 blocks, then projected back to ``d_model``. Token count is preserved
so the SOFTS projection head is unchanged.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _trunc_normal_(tensor, std=0.02):
    nn.init.trunc_normal_(tensor, std=std)


class DINOv3MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DINOv3Attention(nn.Module):
    """Multi-head self-attention with DINOv3 RoPE on patch tokens only."""

    def __init__(self, dim, num_heads, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embed dim {dim} không chia hết cho num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        # Separate Q/K/V so the names line up with Hugging Face DINOv3ViTAttention
        # (q_proj / k_proj / v_proj / o_proj) when loading pretrained weights.
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.o_proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x, rope, attn_mask=None):
        # x: [B, N, C]   rope: (sin, cos) each [N_patches, head_dim]
        b, n, _ = x.shape
        q = self.q_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        if rope is not None:
            q, k = apply_rotary_pos_emb(q, k, rope[0], rope[1])

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        out = self.proj_drop(self.o_proj(out))
        return out, attn


class DINOv3Block(nn.Module):
    """Pre-norm block = official SelfAttentionBlock / HF DINOv3ViTLayer."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, layerscale_init=1e-5, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DINOv3Attention(dim, num_heads, proj_dropout=dropout)
        self.ls1 = nn.Parameter(layerscale_init * torch.ones(dim))
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = DINOv3MLP(dim, int(dim * mlp_ratio), dropout=dropout)
        self.ls2 = nn.Parameter(layerscale_init * torch.ones(dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, rope, attn_mask=None):
        attn_out, attn = self.attn(self.norm1(x), rope, attn_mask)
        x = x + self.drop(self.ls1 * attn_out)
        x = x + self.drop(self.ls2 * self.mlp(self.norm2(x)))
        return x, attn


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, sin, cos):
    """RoPE on patch tokens only. Prefix (CLS + registers) is left untouched.

    q, k: [B, heads, N, head_dim]
    sin, cos: [N_patches, head_dim]  — length is the patch count, not N.
    """
    num_tokens = q.shape[-2]
    num_patches = sin.shape[-2]
    num_prefix = num_tokens - num_patches
    if num_prefix < 0:
        raise RuntimeError(
            f"RoPE patch count {num_patches} lớn hơn độ dài chuỗi {num_tokens}"
        )

    # Broadcast [N_patches, head_dim] onto [B, heads, N_patches, head_dim].
    sin = sin[None, None, :, :]
    cos = cos[None, None, :, :]

    q_prefix, q_patches = q.split((num_prefix, num_patches), dim=-2)
    k_prefix, k_patches = k.split((num_prefix, num_patches), dim=-2)
    q_patches = (q_patches * cos) + (rotate_half(q_patches) * sin)
    k_patches = (k_patches * cos) + (rotate_half(k_patches) * sin)
    q = torch.cat((q_prefix, q_patches), dim=-2)
    k = torch.cat((k_prefix, k_patches), dim=-2)
    return q, k


class DINOv3Rope(nn.Module):
    """2D RoPE of DINOv3 ViT-S/16.

    ``normalize_coords='separate'``, ``base=100``.
    ``rescale_coords=2`` is the training-time log-uniform augmentation used by
    ``dinov3_vits16``; it is disabled in eval, matching the official code.
    """

    def __init__(self, head_dim, base=100.0, rescale_coords=2.0):
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(f"head_dim={head_dim} phải chia hết cho 4 để dùng RoPE 2D")
        self.base = base
        self.rescale_coords = rescale_coords
        # periods[i] = base ** (i / (head_dim/4)); angles = 2*pi*coord / periods
        n_freq = head_dim // 4
        exponents = torch.arange(n_freq, dtype=torch.float32) / n_freq
        periods = base ** exponents
        self.register_buffer("periods", periods, persistent=False)

    def forward(self, height, width, device, dtype):
        periods = self.periods.to(device=device, dtype=torch.float32)
        coords_h = torch.arange(0.5, height, device=device, dtype=torch.float32) / height
        coords_w = torch.arange(0.5, width, device=device, dtype=torch.float32) / width
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = coords.flatten(0, 1)
        coords = 2.0 * coords - 1.0  # [-1, 1]

        if self.training and self.rescale_coords is not None:
            rescale_max = math.log(self.rescale_coords)
            rescale = torch.empty(1, device=device, dtype=torch.float32).uniform_(-rescale_max, rescale_max).exp()
            coords = coords * rescale

        # [HW, 2, D/4] -> [HW, D/2] -> [HW, D]
        angles = 2 * math.pi * coords[:, :, None] / periods[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        sin = torch.sin(angles).to(dtype=dtype)
        cos = torch.cos(angles).to(dtype=dtype)
        return sin, cos


class DINOv3Encoder(nn.Module):
    """Drop-in for ``layers.Transformer_EncDec.Encoder``.

    ``forward(x, attn_mask=None, **kwargs) -> (x, attns)`` with
    ``x`` shaped ``[B, L, d_model]``, same length in and out.
    """

    def __init__(
        self,
        d_model,
        depth=12,
        embed_dim=384,
        num_heads=6,
        mlp_ratio=4.0,
        n_register_tokens=4,
        layerscale_init=1e-5,
        dropout=0.0,
        rope_base=100.0,
        pretrained=False,
        pretrained_model_id="facebook/dinov3-vits16-pretrain-lvd1689m",
    ):
        super().__init__()
        self.d_model = d_model
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.n_register_tokens = n_register_tokens
        self.n_prefix = 1 + n_register_tokens

        self.in_proj = nn.Linear(d_model, embed_dim)
        self.out_proj = nn.Linear(embed_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, n_register_tokens, embed_dim))
        self.rope = DINOv3Rope(embed_dim // num_heads, base=rope_base, rescale_coords=2.0)
        self.blocks = nn.ModuleList(
            [
                DINOv3Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    layerscale_init=layerscale_init,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

        if pretrained:
            self.load_pretrained(pretrained_model_id)

    def _init_weights(self):
        _trunc_normal_(self.cls_token, std=0.02)
        _trunc_normal_(self.register_tokens, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                _trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def load_pretrained(self, model_id="facebook/dinov3-vits16-pretrain-lvd1689m"):
        """Copy transformer weights from a Hugging Face DINOv3 ViT checkpoint.

        The image patch embedding is intentionally not loaded: this encoder
        consumes SOFTS series tokens, not pixels. ``in_proj`` / ``out_proj``
        stay randomly initialized (they adapt ``d_model`` <-> 384).
        Returns True on success.
        """
        if self.embed_dim != 384 or self.num_heads != 6:
            print(
                "[DINOv3] Bỏ qua pretrained: checkpoint ViT-S/16 cần "
                f"embed_dim=384, num_heads=6 (đang là {self.embed_dim}, {self.num_heads})."
            )
            return False
        try:
            from transformers import DINOv3ViTModel
        except ImportError:
            print(
                "[DINOv3] Chưa cài `transformers` (bản có DINOv3). "
                "Giữ khởi tạo ngẫu nhiên. Cài: pip install -U transformers"
            )
            return False

        print(f"[DINOv3] Đang tải encoder pretrained: {model_id}")
        hf = DINOv3ViTModel.from_pretrained(model_id)
        src = hf.state_dict()
        mapped = {}
        mapped["cls_token"] = src["embeddings.cls_token"]
        mapped["register_tokens"] = src["embeddings.register_tokens"]
        mapped["norm.weight"] = src["norm.weight"]
        mapped["norm.bias"] = src["norm.bias"]

        n_src_layers = hf.config.num_hidden_layers
        n_copy = min(self.depth, n_src_layers)
        if self.depth != n_src_layers:
            print(
                f"[DINOv3] depth={self.depth} khác checkpoint ({n_src_layers}). "
                f"Chỉ copy {n_copy} block đầu."
            )
        for i in range(n_copy):
            p = f"model.layer.{i}."
            q = f"blocks.{i}."
            mapped[q + "norm1.weight"] = src[p + "norm1.weight"]
            mapped[q + "norm1.bias"] = src[p + "norm1.bias"]
            mapped[q + "norm2.weight"] = src[p + "norm2.weight"]
            mapped[q + "norm2.bias"] = src[p + "norm2.bias"]
            mapped[q + "ls1"] = src[p + "layer_scale1.lambda1"]
            mapped[q + "ls2"] = src[p + "layer_scale2.lambda1"]
            mapped[q + "attn.q_proj.weight"] = src[p + "attention.q_proj.weight"]
            mapped[q + "attn.q_proj.bias"] = src[p + "attention.q_proj.bias"]
            mapped[q + "attn.k_proj.weight"] = src[p + "attention.k_proj.weight"]
            mapped[q + "attn.k_proj.bias"] = src[p + "attention.k_proj.bias"]
            mapped[q + "attn.v_proj.weight"] = src[p + "attention.v_proj.weight"]
            mapped[q + "attn.v_proj.bias"] = src[p + "attention.v_proj.bias"]
            mapped[q + "attn.o_proj.weight"] = src[p + "attention.o_proj.weight"]
            mapped[q + "attn.o_proj.bias"] = src[p + "attention.o_proj.bias"]
            mapped[q + "mlp.fc1.weight"] = src[p + "mlp.up_proj.weight"]
            mapped[q + "mlp.fc1.bias"] = src[p + "mlp.up_proj.bias"]
            mapped[q + "mlp.fc2.weight"] = src[p + "mlp.down_proj.weight"]
            mapped[q + "mlp.fc2.bias"] = src[p + "mlp.down_proj.bias"]

        missing, unexpected = self.load_state_dict(mapped, strict=False)
        # in_proj / out_proj are supposed to stay missing.
        real_missing = [k for k in missing if not k.startswith(("in_proj", "out_proj"))]
        if real_missing:
            print(f"[DINOv3] Thiếu key sau khi nạp: {real_missing}")
        print(
            f"[DINOv3] Đã nạp {n_copy} block. "
            f"Adapter in_proj/out_proj giữ khởi tạo ngẫu nhiên "
            f"(unexpected bỏ qua: {len(unexpected)})."
        )
        del hf
        return True

    def _grid(self, length):
        height = int(math.ceil(math.sqrt(length)))
        width = int(math.ceil(length / height))
        return height, width

    def forward(self, x, attn_mask=None, tau=None, delta=None, **kwargs):
        """
        x: [B, L, d_model] series tokens from DataEmbedding_inverted.
        attn_mask / tau / delta are accepted so the call site matches
        ``Encoder.forward``; they are not used (DINOv3 has its own RoPE).
        """
        del attn_mask, tau, delta, kwargs
        b, length, _ = x.shape
        tokens = self.in_proj(x)

        height, width = self._grid(length)
        n_patches = height * width
        n_pad = n_patches - length
        if n_pad:
            pad = tokens.new_zeros(b, n_pad, self.embed_dim)
            tokens = torch.cat([tokens, pad], dim=1)

        cls = self.cls_token.expand(b, -1, -1)
        registers = self.register_tokens.expand(b, -1, -1)
        hidden = torch.cat([cls, registers, tokens], dim=1)

        rope = self.rope(height, width, device=hidden.device, dtype=hidden.dtype)

        key_mask = None
        if n_pad:
            # Ban the padded patch positions from being attended to.
            total = self.n_prefix + n_patches
            allow = hidden.new_ones(b, total)
            allow[:, self.n_prefix + length :] = 0
            key_mask = torch.zeros(b, 1, 1, total, device=hidden.device, dtype=hidden.dtype)
            key_mask = key_mask.masked_fill(allow[:, None, None, :] == 0, float("-inf"))

        attns = []
        for block in self.blocks:
            hidden, attn = block(hidden, rope, key_mask)
            attns.append(attn)

        hidden = self.norm(hidden)
        patch = hidden[:, self.n_prefix : self.n_prefix + length, :]
        out = self.out_proj(patch)
        return out, attns
