"""
# encoding: utf-8
#!/usr/bin/env python3

@Author : ZDZ
@Time : 2023/8/19 23:00 
"""

import math
import torch
from torch import nn
from functools import partial
from collections import OrderedDict
from torchvision import transforms


class DropPath(nn.Module):
    '''
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    '''

    def __init__(self, drop_prob: float = 0., training: bool = False):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.training = training

    def forward(self, x):
        '''
        Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
        This is the same as the DropConnect impl I created for EfficientNet, etc. networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.
        '''

        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class PatchEmbed(nn.Module):
    '''
    2D Image to Patch Embedding
    '''

    def __init__(self, img_size=256, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patchs = self.grid_size ** 2

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim,
                 num_heads=8,
                 qkv_bias=False,
                 s_bias=False,
                 gate_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.s = nn.Linear(dim, dim, bias=s_bias)
        self.gate = nn.Linear(dim, dim, bias=gate_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x, e):
        # [batch_size, num_patches, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        gate = self.gate(x)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # s(): -> [batch_size, num_patchs, total_embed_dim]
        # reshape: -> [batch_size, num_patches, num_heads, embed_dim_per_head]
        # permute: -> [batch_size, num_heads, num_patches, embed_dim_per_head]
        s = self.s(e).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches]
        # @: multiply -> [batch_size, num_heads, num_patches, num_patches]
        attn = (q @ (k + s).transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x * gate
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 s_bias=False,
                 gate_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, s_bias=s_bias, gate_bias=gate_bias,
                              qk_scale=qk_scale, attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio)
        self.norm3 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop_ratio)

    def forward(self, x, e):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm2(e)))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x


class ViTEncoder(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_c=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, s_bias=True, gate_bias=True, qk_scale=None, drop_ratio=0., attn_drop_ratio=0.,
                 drop_path_ratio=0.,embed_layer=PatchEmbed, norm_layer=None, act_layer=None):
        '''
        :param img_size: input image size
        :param patch_size: patch size
        :param in_c: number of input channels
        :param embed_dim: embedding dimension
        :param depth: depth of transformer
        :param num_heads: number of attention heads
        :param mlp_ratio: ratio of mlp hidden dim to embedding dim
        :param qkv_bias: enable bias for qkv if True
        :param s_bias: enable bias for s if True
        :param gate_bias: enable bias for gate if True
        :param qk_scale: override default qk scale of head_dim ** -0.5 if set
        :param drop_ratio: dropout rate
        :param attn_drop_ratio: attention dropout rate
        :param drop_path_ratio: stochastic depth rate
        :param embed_layer: patch embedding layer
        :param norm_layer: normalization layer
        :param act_layer: activate function
        '''
        super(ViTEncoder, self).__init__()
        self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        num_patchs = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patchs, embed_dim))
        self.pos_drop = nn.Dropout(drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, s_bias=s_bias,
                  gate_bias=gate_bias, qk_scale=qk_scale, drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
                  drop_path_ratio=dpr[i], norm_layer=norm_layer, act_layer=act_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, e):
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = x + block(x, e)
        x = self.norm(x)
        return x


class DownSample(nn.Module):
    def __init__(self, embed_dim=48, grid_size=64):
        super(DownSample, self).__init__()
        self.grid_size = grid_size
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim * 2, kernel_size=3, stride=2, padding=1,
                      bias=False),
            nn.InstanceNorm2d(num_features=embed_dim * 2, track_running_stats=False),
            nn.GELU())

    def forward(self, x):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, self.grid_size, self.grid_size)
        x = self.downsample(x).flatten(2).transpose(1, 2)
        return x


class UpSample(nn.Module):
    def __init__(self, embed_dim=64, grid_size=64):
        super(UpSample, self).__init__()
        self.grid_size = grid_size
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim // 2, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.InstanceNorm2d(num_features=embed_dim // 2, track_running_stats=False),
            nn.GELU())

    def forward(self, x):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, self.grid_size, self.grid_size)
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        x = self.upsample(x).flatten(2).transpose(1, 2)
        return x


class PixelShuffleLast(nn.Module):
    def __init__(self, in_c, out_c, grid_size, scale_factor=4):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(PixelShuffleLast, self).__init__()
        self.grid_size = grid_size
        self.block1 = nn.Sequential(
            nn.Conv2d(in_c, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, out_c, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, self.grid_size, self.grid_size)
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
