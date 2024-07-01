"""
# encoding: utf-8
#!/usr/bin/env python3

@Author : ZDZ
@Time : 2023/8/22 20:05 
"""

import sys
import time
import torch
import datetime
from torch import nn
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from options import Options
from dataset import SatelliteDateset
from torch.utils.data import DataLoader
from loss import StyleLoss, PerceptualLoss
from models import Generator, Discriminator
from utils import psnr_value, ssim_value, mape_value, tensor_to_image


class Trainer(nn.Module):
    def __init__(self, batch_size, epoch, n_epoch, lr, beta1, beta2, img_size, in_c, out_c,
                 patch_size, embed_dim, depth, num_heads, adv_loss_weight, per_loss_weight,
                 sty_loss_weight, l1_loss_weight, sobel_loss_weight, save_model_root, train_result_root,
                 sample_interval, drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.):
        '''
        :param batch_size: batch size for training
        :param epoch: epoch to start training from
        :param n_epoch: number of epochs of training
        :param lr: adam learning rate
        :param beta1: adam decay of first order momentum of gradient
        :param beta2: adam decay of second order momentum of gradient
        :param img_size: input image size
        :param in_c: input image channels
        :param out_c: out image channels
        :param patch_size: image size after patch embedding
        :param embed_dim: dimensions of each patch after patch embedding
        :param depth: vision transformer block depth
        :param numheads: number of heads of multi-head attention
        :param adv_loss_weight: adversarial loss weight
        :param per_loss_weight: perceptual loss weight
        :param sty_loss_weight: style loss weight
        :param l1_loss_weight: l1 loss weight
        :param sobel_loss_weight: sobel loss weight
        :param save_model_root: The location where the model is saved
        :param train_result_root: The location where the train result is saved
        :param sample_interval: interval between saving image samples
        :param drop_ratio: dropout rate
        :param attn_drop_ratio: attention dropout rate
        :param drop_path_ratio: stochastic depth rate
        '''
        super(Trainer, self).__init__()

        self.batch_size = batch_size
        self.epoch = epoch
        self.n_epoch = n_epoch
        self.writer = SummaryWriter(log_dir='./logs')
        self.save_model_root = save_model_root
        self.train_result_root = train_result_root

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.G = Generator(img_size=img_size, in_c=in_c, out_c=out_c, patch_size=patch_size, embed_dim=embed_dim,
                           depth=depth, num_heads=num_heads, drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
                           drop_path_ratio=drop_path_ratio).to(self.device)
        self.D = Discriminator(in_channels=out_c).to(self.device)

        self.l1_loss = nn.L1Loss().to(self.device)
        self.l2_loss = nn.MSELoss().to(self.device)
        self.sty_loss = StyleLoss().to(self.device)
        self.per_loss = PerceptualLoss().to(self.device)

        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=1e-5)
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=lr * 0.01, betas=(beta1, beta2), weight_decay=1e-5)

        self.adv_loss_weight = adv_loss_weight
        self.per_loss_weight = per_loss_weight
        self.sty_loss_weight = sty_loss_weight
        self.l1_loss_weight = l1_loss_weight
        self.sobel_loss_weight = sobel_loss_weight
        self.sample_interval = sample_interval

    def load_net(self, epoch):
        if epoch > 0:
            self.G.load_state_dict(torch.load(self.save_model_root + 'G_' + str(epoch) + '.pth'))
            self.D.load_state_dict(torch.load(self.save_model_root + 'D_' + str(epoch) + '.pth'))

    def save_net(self, epoch):
        torch.save(self.G.state_dict(), self.save_model_root + 'G_' + str(epoch) + '.pth')
        torch.save(self.D.state_dict(), self.save_model_root + 'D_' + str(epoch) + '.pth')

    def forward(self, dataset):
        start_time, prev_time = time.time(), time.time()
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.load_net(self.epoch - 1)
        for epoch in range(self.epoch, self.n_epoch):
            for index, x in enumerate(dataloader):
                img_truth, mask, sobel_mask, sobel = (x[0].to(self.device), x[1].to(self.device),
                                                      x[2].to(self.device), x[3].to(self.device))
                img = nn.Parameter(img_truth * (1 - mask), requires_grad=True)
                sobel_mask = nn.Parameter(sobel_mask, requires_grad=True)

                valid = nn.Parameter(torch.ones([self.batch_size, 1, 30, 30]).to(self.device), requires_grad=False)
                fake = nn.Parameter(torch.zeros([self.batch_size, 1, 30, 30]).to(self.device), requires_grad=False)

                self.opt_G.zero_grad()
                img_fake, sobel_fake = self.G(img, sobel_mask, mask)

                img_fake_dis, _ = self.D(img_fake)

                adv_loss = self.l1_loss(img_fake_dis, valid) * self.adv_loss_weight

                per_loss = self.per_loss(img_fake, img_truth) * self.per_loss_weight

                sty_loss = self.sty_loss(img_fake, img_truth) * self.sty_loss_weight

                l1_loss = self.l1_loss(img_fake, img_truth) * self.l1_loss_weight

                sobel_loss = self.l1_loss(sobel_fake, sobel) * self.sobel_loss_weight

                G_loss = (adv_loss + per_loss + sty_loss + l1_loss + sobel_loss) / 5

                G_loss.backward()
                self.opt_G.step()

                self.opt_D.zero_grad()
                img_fake_dis, _ = self.D(img_fake.detach())
                img_truth_dis, _ = self.D(img_truth)
                dis_real_loss = self.l1_loss(img_truth_dis, valid)
                dis_fake_loss = self.l1_loss(img_fake_dis, fake)
                D_loss = (dis_real_loss + dis_fake_loss) / 2

                D_loss.backward()
                self.opt_D.step()

                psnr = psnr_value(img_fake, img_truth, self.device)
                ssim = ssim_value(img_fake, img_truth, self.device)
                mape = mape_value(img_fake, img_truth, self.device)

                # print the Loss by time
                batches_done = self.epoch * len(dataloader) + index + 1
                batches_left = self.n_epoch * len(dataloader) - batches_done
                time_use = datetime.timedelta(seconds=time.time() - start_time)
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                sys.stdout.write(
                    "\033[36m[Epoch %d/%d][Batch %2d/%d]\033[33m"
                    "\033[33m[G LOSS: %.4f][D LOSS: %.4f]\033[0m"
                    "\033[31m[ADV: %.4f PER: %.4f STY: %.4f L1: %.4f SOBEL: %.4F DIS_REAL: %.4f DIS_FAKE: %.4f]\033[0m"
                    "\033[36m[PSNR: %.4f SSIM: %.4f MAPE: %.4f]\033[33m"
                    " \033[32mETA: %s\033[0m TAS: %s \n"
                    % (epoch, self.n_epoch, index, len(dataloader), G_loss.item(), D_loss.item(),
                       adv_loss.item(), per_loss.item(), sty_loss.item(), l1_loss.item(), sobel_loss.item(),
                       dis_real_loss.item(), dis_fake_loss.item(), psnr, ssim, mape, time_left, time_use))

                # Every n batches, save the inpainted image
                batches_done = epoch * len(dataloader) + index + 1
                if batches_done % self.sample_interval == 0:

                    real_img = torch.split(img_truth, split_size_or_sections=1, dim=0)
                    real_img_comp = torch.cat(real_img, dim=3).reshape(3, 256, -1)

                    sobel_img = torch.split(sobel_mask, split_size_or_sections=1, dim=0)
                    sobel_img_comp = torch.cat(sobel_img, dim=3).reshape(1, 256, -1).repeat(3, 1, 1)

                    real_miss = torch.split(img + mask, split_size_or_sections=1, dim=0)
                    real_miss_comp = torch.cat(real_miss, dim=3).reshape(3, 256, -1)

                    fake_sobel = torch.split(sobel_fake, split_size_or_sections=1, dim=0)
                    fake_sobel_comp = torch.cat(fake_sobel, dim=3).reshape(1, 256, -1).repeat(3, 1, 1)

                    fake_img = torch.split(img_fake, split_size_or_sections=1, dim=0)
                    fake_img_comp = torch.cat(fake_img, dim=3).reshape(3, 256, -1)

                    comp = torch.cat([real_img_comp, sobel_img_comp, real_miss_comp,
                                      fake_sobel_comp, fake_img_comp], dim=1)
                    comp_pic = tensor_to_image(comp)
                    comp_pic.save(self.train_result_root + str(epoch) + '_' + str(index) + '.png')

                    grid_image = make_grid([real_img_comp, sobel_img_comp, real_miss_comp,
                                            fake_sobel_comp, fake_img_comp], nrow=1)
                    self.writer.add_image('Original / Sobel / Corrupted / Sobel Reconstructed / Reconstructed',
                                          grid_image, batches_done)

                    self.writer.add_scalars('LOSS', {'G LOSS': G_loss.item(),
                                                     'D LOSS': D_loss.item()}, batches_done)

                    self.writer.add_scalars('G LOSS DETAIL', {'ADV LOSS': adv_loss.item(),
                                                              'PER LOSS': per_loss.item(), 'STY LOSS': sty_loss.item(),
                                                              'L1 LOSS': l1_loss.item(),
                                                              'SOBEL LOSS': sobel_loss.item()}, batches_done)

                    self.writer.add_scalars('D LOSS DETAIL', {'DIS REAL': dis_real_loss.item(),
                                                              'DIS FAKE': dis_fake_loss.item()}, batches_done)

                    self.writer.add_scalars('EVAL', {'PSNR': psnr / 100, 'SSIM': ssim,
                                                     'MAPE': mape}, batches_done)

            self.save_net(epoch)

        return self.G


if __name__ == '__main__':
    opt = Options()
    train = Trainer(batch_size=opt.BATCH_SIZE,
                    epoch=opt.EPOCH, n_epoch=opt.N_EPOCH, lr=opt.LR, beta1=opt.BETA1, beta2=opt.BETA2,
                    img_size=opt.IMG_SIZE, in_c=opt.IN_C, out_c=opt.OUT_C, patch_size=opt.PATCH_SIZE,
                    embed_dim=opt.EMBED_DIM, depth=opt.DEPTH, num_heads=opt.NUM_HEADS,
                    adv_loss_weight=opt.ADV_LOSS_WEIGHT, per_loss_weight=opt.PER_LOSS_WEIGHT,
                    sty_loss_weight=opt.STY_LOSS_WEIGHT, l1_loss_weight=opt.L1_LOSS_WEIGHT,
                    sobel_loss_weight=opt.SOBEL_LOSS_WEIGHT, sample_interval=opt.SAMPLE_INTERVAL,
                    save_model_root=opt.SAVE_MODEL_ROOT, train_result_root=opt.TRAIN_RESULT_ROOT,
                    drop_ratio=0.2, attn_drop_ratio=0.3, drop_path_ratio=0.2)

    dataset = SatelliteDateset(image_root=opt.TRAIN_IMG_ROOT, mask_root=opt.TRAIN_MASK_ROOT)
    G = train(dataset)