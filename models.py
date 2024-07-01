"""
# encoding: utf-8
#!/usr/bin/env python3

@Author : ZDZ
@Time : 2023/12/18 13:42 
"""

import torch
from torch import nn
from networks import ViTEncoder, PatchEmbed, DownSample, UpSample, PixelShuffleLast, ResidualBlock


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class Generator(BaseNetwork):
    def __init__(self, img_size=256, in_c=4, out_c=3, patch_size=4,
                 embed_dim=64, depth=[1, 2, 3, 4], num_heads=[1, 2, 4, 8],
                 drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0., init_weights=True):
        super(Generator, self).__init__()
        self.grid_size = img_size // patch_size
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        self.patch_embed_s = PatchEmbed(img_size=img_size, patch_size=patch_size, in_c=1, embed_dim=embed_dim)
        self.encoder64 = ViTEncoder(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim,
                                    depth=depth[0], num_heads=num_heads[0], drop_ratio=drop_ratio,
                                    attn_drop_ratio=attn_drop_ratio, drop_path_ratio=drop_path_ratio)

        self.downsample32 = DownSample(embed_dim=embed_dim, grid_size=self.grid_size)
        self.downsample32_s = DownSample(embed_dim=embed_dim, grid_size=self.grid_size)
        self.encoder32 = ViTEncoder(img_size=img_size, patch_size=patch_size * 2, in_c=in_c, embed_dim=embed_dim * 2,
                                    depth=depth[1], num_heads=num_heads[1], drop_ratio=drop_ratio,
                                    attn_drop_ratio=attn_drop_ratio, drop_path_ratio=drop_path_ratio)

        self.downsample16 = DownSample(embed_dim=embed_dim * 2, grid_size=self.grid_size // 2)
        self.downsample16_s = DownSample(embed_dim=embed_dim * 2, grid_size=self.grid_size // 2)
        self.encoder16 = ViTEncoder(img_size=img_size, patch_size=patch_size * 4, in_c=in_c, embed_dim=embed_dim * 4,
                                    depth=depth[2], num_heads=num_heads[2], drop_ratio=drop_ratio,
                                    attn_drop_ratio=attn_drop_ratio, drop_path_ratio=drop_path_ratio)

        self.downsample8 = DownSample(embed_dim=embed_dim * 4, grid_size=self.grid_size // 4)
        self.downsample8_s = DownSample(embed_dim=embed_dim * 4, grid_size=self.grid_size // 4)
        self.encoder8 = ViTEncoder(img_size=img_size, patch_size=patch_size * 8, in_c=in_c, embed_dim=embed_dim * 8,
                                   depth=depth[3], num_heads=num_heads[3], drop_ratio=drop_ratio,
                                   attn_drop_ratio=attn_drop_ratio, drop_path_ratio=drop_path_ratio)

        self.res_block1 = ResidualBlock(embed_dim * 8)
        self.res_block2 = ResidualBlock(embed_dim * 8)
        self.res_block3 = ResidualBlock(embed_dim * 8)
        self.res_block4 = ResidualBlock(embed_dim * 8)
        self.res_block5 = ResidualBlock(embed_dim * 8)
        self.res_block6 = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim * 8, out_channels=embed_dim * 8, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.InstanceNorm2d(num_features=embed_dim * 8, track_running_stats=False),
            nn.GELU())

        self.upsample16 = UpSample(embed_dim=embed_dim * 8, grid_size=self.grid_size // 8)
        self.upsample16_s = UpSample(embed_dim=embed_dim * 8, grid_size=self.grid_size // 8)
        self.concat16 = nn.Conv1d(in_channels=(self.grid_size // 4) ** 2 * 2, out_channels=(self.grid_size // 4) ** 2,
                                  kernel_size=1, stride=1)
        self.decoder16 = ViTEncoder(img_size=img_size, patch_size=patch_size * 4, in_c=in_c, embed_dim=embed_dim * 4,
                                    depth=depth[2], num_heads=num_heads[2], drop_ratio=drop_ratio,
                                    attn_drop_ratio=attn_drop_ratio, drop_path_ratio=drop_path_ratio)

        self.upsample32 = UpSample(embed_dim=embed_dim * 4, grid_size=self.grid_size // 4)
        self.upsample32_s = UpSample(embed_dim=embed_dim * 4, grid_size=self.grid_size // 4)
        self.concat32 = nn.Conv1d(in_channels=(self.grid_size // 2) ** 2 * 2, out_channels=(self.grid_size // 2) ** 2,
                                  kernel_size=1, stride=1)
        self.decoder32 = ViTEncoder(img_size=img_size, patch_size=patch_size * 2, in_c=in_c, embed_dim=embed_dim * 2,
                                    depth=depth[1], num_heads=num_heads[1], drop_ratio=drop_ratio,
                                    attn_drop_ratio=attn_drop_ratio, drop_path_ratio=drop_path_ratio)

        self.upsample64 = UpSample(embed_dim=embed_dim * 2, grid_size=self.grid_size // 2)
        self.upsample64_s = UpSample(embed_dim=embed_dim * 2, grid_size=self.grid_size // 2)
        self.concat64 = nn.Conv1d(in_channels=self.grid_size ** 2 * 2, out_channels=self.grid_size ** 2,
                                  kernel_size=1, stride=1)
        self.decoder64 = ViTEncoder(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim,
                                    depth=depth[0], num_heads=num_heads[0], drop_ratio=drop_ratio,
                                    attn_drop_ratio=attn_drop_ratio, drop_path_ratio=drop_path_ratio)

        self.out_s = PixelShuffleLast(in_c=embed_dim, out_c=1, grid_size=self.grid_size)
        self.out_s_patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_c=1, embed_dim=embed_dim)
        self.out_concat = nn.Conv1d(in_channels=self.grid_size ** 2 * 2, out_channels=self.grid_size ** 2,
                                  kernel_size=1, stride=1)
        self.out = PixelShuffleLast(in_c=embed_dim, out_c=out_c, grid_size=self.grid_size)


        if init_weights:
            self.init_weights()

    def forward(self, x, e, mask=None):
        x = torch.cat([x, mask], dim=1) if mask is not None else x

        encoder64 = self.patch_embed(x)
        encoder64_s = self.patch_embed_s(e)
        encoder64 = self.encoder64(encoder64, encoder64_s)

        encoder32 = self.downsample32(encoder64)
        encoder32_s = self.downsample32_s(encoder64_s)
        encoder32 = self.encoder32(encoder32, encoder32_s)

        encoder16 = self.downsample16(encoder32)
        encoder16_s = self.downsample16_s(encoder32_s)
        encoder16 = self.encoder16(encoder16, encoder16_s)

        encoder8 = self.downsample8(encoder16)
        encoder8_s = self.downsample8_s(encoder16_s)
        encoder8 = self.encoder8(encoder8, encoder8_s)

        B, N, C = encoder8_s.shape
        middle_s = encoder8_s.transpose(1, 2).view(B, C, self.grid_size // 8, self.grid_size // 8)
        middle_s = self.res_block1(middle_s)
        middle_s = self.res_block2(middle_s)
        middle_s = self.res_block3(middle_s)
        middle_s = self.res_block4(middle_s)
        middle_s = self.res_block5(middle_s)
        middle_s = self.res_block6(middle_s)
        middle_s = middle_s.flatten(2).transpose(1, 2)

        decoder16 = self.upsample16(encoder8)
        decoder16_s = self.upsample16_s(middle_s + encoder8_s)
        decoder16 = self.concat16(torch.cat([encoder16, decoder16], dim=1))
        decoder16 = self.decoder16(decoder16, decoder16_s)

        decoder32 = self.upsample32(decoder16)
        decoder32_s = self.upsample32_s(decoder16_s + encoder16_s)
        decoder32 = self.concat32(torch.cat([encoder32, decoder32], dim=1))
        decoder32 = self.decoder32(decoder32, decoder32_s)

        decoder64 = self.upsample64(decoder32)
        decoder64_s = self.upsample64_s(decoder32_s + encoder32_s)
        decoder64 = self.concat64(torch.cat([encoder64, decoder64], dim=1))
        decoder64 = self.decoder64(decoder64, decoder64_s)

        out_s = torch.tanh(self.out_s(decoder64_s))
        out_s_patched = self.out_s_patch_embed(out_s)
        out = self.out_concat(torch.cat([decoder64, out_s_patched], dim=1))
        out = torch.tanh(self.out(out))

        return out, out_s


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]
