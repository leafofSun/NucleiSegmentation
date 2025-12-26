# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type
from .common import LayerNorm2d, MLPBlock

try:
    import clip as _clip
except Exception:
    _clip = None

class Adapter_Layer(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=0.25, norm_layer=nn.LayerNorm, skip_connect=True):
        super().__init__()
        # [关键] 移除 skip_connect 和 norm，Adapter 只输出增量
        hidden_dim = int(embed_dim * mlp_ratio)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim, bias=False),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim, bias=False),
                nn.Sigmoid()
        )

        self.spatial = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.ReLU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1, bias=False),
                # [关键] 移除末端 ReLU，保持线性，允许输出负值
        )
        
        # 基础权重初始化 (Kaiming Init)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        # [关键] 零初始化 - 确保初始状态为 Identity Mapping
        # Spatial 分支最后一层权重为 0 -> x_spatial = 0
        nn.init.zeros_(self.spatial[2].weight)
        # Channel 分支：虽然 Sigmoid(0)=0.5，但通过负偏置可以让输出接近 1
        # 或者接受 0.5 的 scaling（影响较小，因为 spatial 输出为 0）
        nn.init.zeros_(self.channel[2].weight)
        # [可选] 让 channel 分支的最后一个 Linear 输出接近 0，使得 Sigmoid 输入为负大数，输出接近 0
        # 但为了简化，我们接受 0.5 的 scaling，因为关键的是 spatial 分支必须为 0
                
    def forward(self, x):
        # x: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0,3,1,2)
        B, C, H_orig, W_orig = x.size()
        
        # Channel Attention
        x_channel = self.channel(self.avg_pool(x).view(B, C)).view(B, C, 1, 1) * x
        
        # Spatial Adapter
        x_spatial = self.spatial(x_channel)
        
        if x_spatial.shape[2] != H_orig or x_spatial.shape[3] != W_orig:
            x_spatial = F.interpolate(x_spatial, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
        
        # [关键修改] 直接返回增量 x_spatial (B, C, H, W)
        # 不要加回 x，也不要做 Norm，让 Block 去处理加法
        
        # (B, C, H, W) -> (B, H, W, C)
        x_spatial = x_spatial.permute(0,2,3,1)
        return x_spatial 
    

class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        adapter_train = False,
        use_clip: bool = False,
        clip_model_name: str = "ViT-B/16",
        clip_pretrained: bool = True
    ) -> None:
        super().__init__()
        self.use_clip = use_clip
        self.clip_model_name = clip_model_name
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.clip = None
        if self.use_clip:
            if _clip is None:
                raise ImportError("clip not installed")
            self.clip, _ = _clip.load(self.clip_model_name, device="cpu", jit=False)

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                adapter = adapter_train,
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_clip:
            visual = self.clip.visual
            x_conv = visual.conv1(x)
            B, C, Hc, Wc = x_conv.shape
            x_seq = x_conv.reshape(B, C, Hc * Wc).permute(0, 2, 1)

            if hasattr(visual, "class_embedding"):
                cls = visual.class_embedding.to(x_seq.dtype)
            else:
                cls = visual.cls_token.to(x_seq.dtype)
            cls = cls.unsqueeze(0) if cls.ndim == 2 else cls
            cls = cls.expand(B, -1, -1)
            x_seq = torch.cat([cls, x_seq], dim=1)

            if hasattr(visual, "positional_embedding"):
                pos_embed = visual.positional_embedding.to(x_seq.dtype)
                x_seq = x_seq + pos_embed

            if hasattr(visual, "ln_pre"):
                x_seq = visual.ln_pre(x_seq)

            x_tr = x_seq.permute(1, 0, 2)
            x_tr = visual.transformer(x_tr)
            x_tr = x_tr.permute(1, 0, 2)

            if hasattr(visual, "ln_post"):
                x_tr = visual.ln_post(x_tr)

            x_patches = x_tr[:, 1:, :].contiguous().view(B, Hc, Wc, -1)
            x = self.neck(x_patches.permute(0, 3, 1, 2))
            return x

        x = self.patch_embed(x)
        if self.pos_embed is not None:
            B, H, W, C = x.shape
            pos_embed = self.pos_embed
            if pos_embed.shape[1] != H or pos_embed.shape[2] != W:
                pos_embed = pos_embed.permute(0, 3, 1, 2)
                pos_embed = F.interpolate(pos_embed, (H, W), mode='bilinear', align_corners=False)
                pos_embed = pos_embed.permute(0, 2, 3, 1)
            x = x + pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, norm_layer=nn.LayerNorm, act_layer=nn.GELU, use_rel_pos=False, rel_pos_zero_init=True, window_size=0, input_size=None, adapter=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.adapter = adapter
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, use_rel_pos=use_rel_pos, rel_pos_zero_init=rel_pos_zero_init, input_size=input_size if window_size == 0 else (window_size, window_size))
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.window_size = window_size
        if self.adapter:
            # [关键] Adapter 不需要 Norm Layer，它只是一个加法分支
            self.Adapter = Adapter_Layer(dim)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        
        # [关键] Parallel Adapter Structure
        if self.adapter:
            x_norm = self.norm2(x)
            # x_new = x + MLP(Norm(x)) + Adapter(Norm(x))
            # Adapter 必须初始化为 0，这样初始状态下 x_new = x + MLP(Norm(x))，完全等同于原始 ViT
            x = x + self.mlp(x_norm) + self.Adapter(x_norm)
        else:
            x = x + self.mlp(self.norm2(x))
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, use_rel_pos=False, rel_pos_zero_init=True, input_size=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x):
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)

def window_unpartition(windows, window_size, pad_hw, hw):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x

def get_rel_pos(q_size, k_size, rel_pos):
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1), size=max_rel_dist, mode="linear")
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]

def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    r_q = r_q.to(Rh.dtype)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).view(B, q_h * q_w, k_h * k_w)
    return attn

class PatchEmbed(nn.Module):
    def __init__(self, kernel_size=(16, 16), stride=(16, 16), padding=(0, 0), in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x