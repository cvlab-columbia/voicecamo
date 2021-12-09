import math
from typing import List, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
import time
import os
import pdb
import Levenshtein as Lev
from torch.nn import CTCLoss
from omegaconf.dictconfig import DictConfig
import numpy as np
import pdb
import librosa
torch.autograd.set_detect_anomaly(True)

from src.decoder import GreedyDecoder
from src.validation import CharErrorRate, WordErrorRate

class DownConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilation=1,
                 norm_fn='bn',
                 act='prelu'):
        super(DownConvBlock, self).__init__()
        pad = (kernel_size - 1) // 2 * dilation
        block = []
        block.append(nn.ReflectionPad2d(pad))
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0, dilation, bias=norm_fn is None))
        if norm_fn == 'bn':
            block.append(nn.BatchNorm2d(out_channels))
        if act == 'prelu':
            block.append(nn.PReLU())
        elif act == 'lrelu':
            block.append(nn.LeakyReLU())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilation=1,
                 norm_fn='bn',
                 act='prelu',
                 up_mode='upconv'):
        super(UpConvBlock, self).__init__()
        pad = (kernel_size - 1) // 2 * dilation
        block = []
        if up_mode == 'upconv':
            block.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, pad, dilation,
                                            bias=norm_fn is None))
        elif up_mode == 'upsample':
            block.append(nn.Upsample(scale_factor=2))
            block.append(nn.ReflectionPad2d(pad))
            block.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride, 0, dilation,
                                   bias=norm_fn is None))
        if norm_fn == 'bn':
            block.append(nn.BatchNorm2d(out_channels))
        if act == 'prelu':
            block.append(nn.PReLU())
        elif act == 'lrelu':
            block.append(nn.LeakyReLU())
        elif act == 'tanh':
            block.append(nn.Tanh())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

class HalfSecNet(nn.Module):
    def __init__(self):
        super(HalfSecNet, self).__init__()
        ch1 = 64
        ch2 = 128
        ch3 = 128
        self.down1 = nn.Sequential(
            DownConvBlock(2, ch1, 5, 1),
        )
        self.down2 = nn.Sequential(
            DownConvBlock(ch1, ch2, 5, (1,2)),
            DownConvBlock(ch2, ch2, 5, 1),
        )
        self.down3 = nn.Sequential(
            DownConvBlock(ch2, ch3, 5, (1,2)), #TO BE EXAMINED
            DownConvBlock(ch3, ch2, 3, 1),
            DownConvBlock(ch2, ch1, 3, 1),
            DownConvBlock(ch1, 2, 3, 1),
        )


    def forward(self, x):

        down1 = self.down1(x)


        #[4, 64, 256, 203]
        down2 = self.down2(down1)

        #torch.Size([4, 128, 128, 102])
        out = self.down3(down2)


        #torch.Size([1, 2, 161, 26])
        return out

class HalfSecNetWav(nn.Module):
    def __init__(self):
        super(HalfSecNetWav, self).__init__()
        ch1 = 64
        ch2 = 128
        ch3 = 256
        self.down1 = nn.Sequential(
            DownConvBlock(2, ch1, 5, (2,2)),
        )
        self.down2 = nn.Sequential(
            DownConvBlock(ch1, ch2, 5, (2,2)),
            DownConvBlock(ch2, ch2, 5, (2,2)),
        )
        self.down3 = nn.Sequential(
            DownConvBlock(ch2, ch3, 5, (2,2)),
            #TO BE EXAMINED
            DownConvBlock(ch3, ch2, 3, (2,2)),
            DownConvBlock(ch2, ch2, 3, (2,2)),
            DownConvBlock(ch2, ch3, 3, (2,2)),
        )
        self.down4 = nn.Sequential(nn.Conv2d(ch3, 512, (2,2), (1,1), 0, (1,1)),nn.LeakyReLU())
        self.up1 = nn.Sequential(nn.ConvTranspose1d(1,64,1,1),nn.LeakyReLU())
        self.up2 = nn.Sequential(nn.ConvTranspose1d(64, 32, 5,2),nn.LeakyReLU())
        self.up3 = nn.Sequential(nn.ConvTranspose1d(32, 16, 5, 2),nn.LeakyReLU())
        self.up4 = nn.Sequential(nn.ConvTranspose1d(16, 1, 5, 2),nn.Tanh())
        #self.up5 = nn.ConvTranspose1d(8, 1, 5, 2)
        self.linear = nn.Sequential(nn.Linear(4117,8000),nn.Tanh())


    def forward(self, x):
        down1 = self.down1(x)

        #[4, 64, 256, 203]
        down2 = self.down2(down1)

        #torch.Size([4, 128, 128, 102])
        down3 = self.down3(down2)

        down4 = self.down4(down3)

        down4 = down4.reshape(x.shape[0],1,512)
        up1 = self.up1(down4)
        up2 = self.up2(up1)

        up3 = self.up3(up2)
        up4 = self.up4(up3)
        up5 = self.linear(up4)
        #up5 = self.up5(up4)


        #torch.Size([4, 64, 256, 204])
        return up5

class HalfSecNetNoBN(nn.Module):
    def __init__(self):
        super(HalfSecNetNoBN, self).__init__()
        ch1 = 64
        ch2 = 256
        ch3 = 512
        self.down1 = nn.Sequential(
            DownConvBlock(2, ch1, 5, 1,norm_fn='nobn'),
        )
        self.down2 = nn.Sequential(
            DownConvBlock(ch1, ch2, 5, (1,2),norm_fn='nobn'),
            DownConvBlock(ch2, ch2, 5, 1,norm_fn='nobn'),
        )
        self.down3 = nn.Sequential(
            DownConvBlock(ch2, ch3, 5, (1,2),norm_fn='nobn'),
            DownConvBlock(ch3, ch2, 3, 1,norm_fn='nobn'),
            DownConvBlock(ch2, ch2, 3, 1,norm_fn='nobn'),
            DownConvBlock(ch2, ch2, 3, 1,norm_fn='nobn'),
        )

    def forward(self, x):

        down1 = self.down1(x)

        #[4, 64, 256, 203]
        down2 = self.down2(down1)

        #torch.Size([4, 128, 128, 102])
        out = self.down3(down2)

        #torch.Size([4, 64, 256, 204])
        return out
import pdb
class HalfSecNetResidual(nn.Module):
    def __init__(self):
        super(HalfSecNetResidual, self).__init__()
        ch1 = 64
        ch2 = 128
        ch3 = 512
        self.down1 = nn.Sequential(DownConvBlock(2, ch1, 5, 1))
        self.down15 = nn.Sequential(DownConvBlock(ch1,ch2,1,(1,2)))
        self.down2 = nn.Sequential(DownConvBlock(ch1, ch2, 5, (1,2)))
        self.down25 = nn.Sequential(DownConvBlock(ch2, ch3, 1, (1, 1)))
        self.down3 = nn.Sequential(DownConvBlock(ch2, ch3, 5, 1))
        self.down4 = nn.Sequential(DownConvBlock(ch3, ch3, 3, 1))
        self.down45 = nn.Sequential(DownConvBlock(ch3, ch2, 1, (1, 2)))
        self.down5 = nn.Sequential(DownConvBlock(ch3, ch2, 3, (1,2)))
        self.down55 = nn.Sequential(DownConvBlock(ch2, ch1, 1, (1, 1)))
        self.down6 = nn.Sequential(DownConvBlock(ch2, ch1, 3, 1))
        self.down7 = nn.Sequential(DownConvBlock(ch1, 2, 3, 1))


    def forward(self, x):
        down1 = self.down1(x)
        down15 = self.down15(down1)

        down2 = self.down2(down1)
        down2 = down2 + down15
        down25 = self.down25(down2)

        down3 = self.down3(down2)
        down3 = down3 + down25

        down4 = self.down4(down3)
        down4 = down4 + down3
        down45 = self.down45(down4)

        down5 = self.down5(down4)
        down5 = down5 + down45
        down55 = self.down55(down5)

        down6 = self.down6(down5)
        down6 = down6 + down55
        down7 = self.down7(down6)

        return down7

class HalfSecNetResidualNoBN(nn.Module):
    def __init__(self):
        super(HalfSecNetResidualNoBN, self).__init__()
        ch1 = 64
        ch2 = 128
        ch3 = 256
        self.down1 = nn.Sequential(DownConvBlock(2, ch1, 5, 1,norm_fn='nobn'))
        self.down15 = nn.Sequential(DownConvBlock(ch1,ch2,1,(1,2),norm_fn='nobn'))
        self.down2 = nn.Sequential(DownConvBlock(ch1, ch2, 5, (1,2),norm_fn='nobn'))
        self.down25 = nn.Sequential(DownConvBlock(ch2, ch3, 1, (1, 1),norm_fn='nobn'))
        self.down3 = nn.Sequential(DownConvBlock(ch2, ch3, 5, 1,norm_fn='nobn'))
        self.down4 = nn.Sequential(DownConvBlock(ch3, ch3, 3, 1,norm_fn='nobn'))
        self.down45 = nn.Sequential(DownConvBlock(ch3, ch2, 1, (1, 2),norm_fn='nobn'))
        self.down5 = nn.Sequential(DownConvBlock(ch3, ch2, 3, (1,2),norm_fn='nobn'))
        self.down55 = nn.Sequential(DownConvBlock(ch2, ch1, 1, (1, 1),norm_fn='nobn'))
        self.down6 = nn.Sequential(DownConvBlock(ch2, ch1, 3, 1,norm_fn='nobn'))
        self.down7 = nn.Sequential(DownConvBlock(ch1, 2, 3, 1,norm_fn='nobn'))


    def forward(self, x):
        down1 = self.down1(x)
        down15 = self.down15(down1)

        down2 = self.down2(down1)
        down2 = down2 + down15
        down25 = self.down25(down2)

        down3 = self.down3(down2)
        down3 = down3 + down25

        down4 = self.down4(down3)
        down4 = down4 + down3
        down45 = self.down45(down4)

        down5 = self.down5(down4)
        down5 = down5 + down45
        down55 = self.down55(down5)

        down6 = self.down6(down5)
        down6 = down6 + down55
        down7 = self.down7(down6)

        return down7

class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class MaskConv(nn.Module):
    def __init__(self, seq_module):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that the
        results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (BxCxDxT)
        :param seq_module: The sequential module containing the conv stack.
        """
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:
            x = module(x)
            mask = torch.BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths


class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return input_


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size, track_running_stats=False)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths)
        x, h = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        return x


class Lookahead(nn.Module):
    # Wang et al 2016 - Lookahead Convolution Layer for Unidirectional Recurrent Neural Networks
    # input shape - sequence, batch, feature - TxNxH
    # output shape - same as input
    def __init__(self, n_features, context):
        super(Lookahead, self).__init__()
        assert context > 0
        self.context = context
        self.n_features = n_features
        self.pad = (0, self.context - 1)
        self.conv = nn.Conv1d(
            self.n_features,
            self.n_features,
            kernel_size=self.context,
            stride=1,
            groups=self.n_features,
            padding=0,
            bias=False
        )

    def forward(self, x):
        x = x.transpose(0, 1).transpose(1, 2)
        x = F.pad(x, pad=self.pad, value=0)
        x = self.conv(x)
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'n_features=' + str(self.n_features) \
               + ', context=' + str(self.context) + ')'

# Networks
##############################################################################
class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size:tuple, dilation:tuple,
                 stride=1,
                 norm_fn='bn',
                 act='relu'):
        super(Conv2dBlock, self).__init__()
        pad = ((kernel_size[0] - 1) // 2 * dilation[0], (kernel_size[1] - 1) // 2 * dilation[1])
        block = []
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, dilation, bias=norm_fn is None))
        if norm_fn == 'bn':
            block.append(nn.BatchNorm2d(out_channels))
        if act == 'relu':
            block.append(nn.ReLU())
        elif act == 'prelu':
            block.append(nn.PReLU())
        elif act == 'lrelu':
            block.append(nn.LeakyReLU())
        elif act == 'tanh':
            block.append(nn.Tanh())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size:tuple,
                 stride: tuple,
                 norm_fn='bn',
                 act='relu'):
        super(Conv3dBlock, self).__init__()
        pad = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2, (kernel_size[2] - 1) // 2)
        block = []
        block.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, pad, bias=norm_fn is None))
        if norm_fn == 'bn':
            block.append(nn.BatchNorm3d(out_channels))
        if act == 'relu':
            block.append(nn.ReLU())
        elif act == 'prelu':
            block.append(nn.PReLU())
        elif act == 'lrelu':
            block.append(nn.LeakyReLU())
        elif act == 'tanh':
            block.append(nn.Tanh())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class AudioVisualNet(nn.Module):
    def __init__(self,
                 freq_bins=256,
                 time_bins=178,
                 nf=96):
        super(AudioVisualNet, self).__init__()

        # video_kernel_sizes = [(5, 7, 7), (5, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (1, 3, 3)]
        # video_strides      = [(1, 2, 2), (1, 1, 1), (1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 3, 3), (1, 3, 3)]
        # self.encoder_video = self.make_video_branch(video_kernel_sizes, video_strides, nf=128, outf=256)

        audio_kernel_sizes = [(1, 7), (7, 1), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5)]
        audio_dilations    = [(1, 1), (1, 1), (1, 1), (2, 1), (4, 1), (8, 1), (16, 1), (32, 1), (1, 1), (2, 2), (4, 4)]
        self.encoder_audio = self.make_audio_branch(audio_kernel_sizes, audio_dilations, nf=48, outf=8)

        self.lstm = nn.LSTM(input_size=8*freq_bins, hidden_size=100, bidirectional=True)
        self.fc1 = nn.Sequential(nn.Linear(200, 100),
                                nn.ReLU(True),
                                nn.Linear(100, 1))

        # self.fc1 = nn.Sequential(nn.Linear(200, 100),
        #                         nn.ReLU(True),
        #                         nn.Linear(100, 1),
        #                         nn.ReLU(True))

        # self.fc2 = nn.Sequential(nn.Linear(50 * time_bins, 50),
        #                         nn.ReLU(True),
        #                         nn.Linear(50, 1),
        #                         )

    def make_video_branch(self, kernel_sizes, strides, nf=256, outf=256):
        encoder_x = []
        for i in range(len(kernel_sizes)):
            if i == 0:
                encoder_x.append(Conv3dBlock(3, nf, kernel_sizes[i], strides[i]))
            else:
                encoder_x.append(Conv3dBlock(nf, nf, kernel_sizes[i], strides[i]))
        encoder_x.append(Conv3dBlock(nf, outf, (1, 1, 1), (1, 1, 1)))
        return nn.Sequential(*encoder_x)

    def make_audio_branch(self, kernel_sizes, dilations, nf=96, outf=8):
        encoder_x = []
        for i in range(len(kernel_sizes)):
            if i == 0:
                encoder_x.append(Conv2dBlock(2, nf, kernel_sizes[i], dilations[i]))
            else:
                encoder_x.append(Conv2dBlock(nf, nf, kernel_sizes[i], dilations[i]))
        encoder_x.append(Conv2dBlock(nf, outf, (1, 1), (1, 1)))
        return nn.Sequential(*encoder_x)

    def forward(self, s, v_num_frames=60):
        f_s = self.encoder_audio(s)
        f_s = f_s.view(f_s.size(0), -1, f_s.size(3)) # (B, C1, T1)
        f_s = F.interpolate(f_s, size=v_num_frames) # (B, C2, T1)

        # f_v = self.encoder_video(v)
        # f_v = torch.mean(f_v, dim=(-2, -1)) # (B, C2, T2)
        # # f_v = F.interpolate(f_v, size=f_v.size(2) / 5) # (B, C2, T1)
        # f_s = F.interpolate(f_s, size=f_v.size(2)) # (B, C2, T1)
        # # print(f_s.shape, f_v.shape)

        # merge = torch.cat([f_s, f_v], dim=1)
        merge = f_s
        merge = merge.permute(2, 0, 1)  # (T1, B, C1+C2)

        # if self.training is True:
        #     self.lstm.flatten_parameters()
        self.lstm.flatten_parameters()
        merge, _ = self.lstm(merge)

        merge = merge.permute(1, 0, 2)# (B, T1, C1+C2)
        merge = self.fc1(merge)
        out = merge.squeeze(2)
        # print(merge.shape)
        # out = self.fc2(merge.view(merge.size(0), -1))
        return out

class InpaintNet(nn.Module):
    def __init__(self):
        super(InpaintNet, self).__init__()
        ch1 = 64
        ch2 = 128
        ch3 = 256
        self.down1 = nn.Sequential(
            DownConvBlock(2, ch1, 5, 1),
        )
        self.down2 = nn.Sequential(
            DownConvBlock(ch1, ch2, 5, 2),
            DownConvBlock(ch2, ch2, 5, 1),
        )
        self.down3 = nn.Sequential(
            DownConvBlock(2, ch1, 5, 1),
        )
        self.down4 = nn.Sequential(
            DownConvBlock(ch1, ch2, 5, 2),
            DownConvBlock(ch2, ch2, 5, 1),
        )
        self.mid = nn.Sequential(
            DownConvBlock(ch2 * 2, ch3, 3, 2),
            DownConvBlock(ch3, ch3, 3, 1),
            DownConvBlock(ch3, ch3, 3, 1, dilation=2),
            DownConvBlock(ch3, ch3, 3, 1, dilation=4),
            DownConvBlock(ch3, ch3, 3, 1, dilation=8),
            DownConvBlock(ch3, ch3, 3, 1, dilation=16),
            DownConvBlock(ch3, ch3, 3, 1),
            DownConvBlock(ch3, ch3, 3, 1),
            UpConvBlock(ch3, ch2, 3, 2),
        )
        self.up1 = nn.Sequential(
            DownConvBlock(ch2 * 2, ch2, 3, 1),
            UpConvBlock(ch2, ch1, 3, 2),
        )
        self.up2 = nn.Sequential(
            DownConvBlock(ch1 * 2, ch1, 3, 1),
            DownConvBlock(ch1, 2, 3, 1, norm_fn=None, act=None)
        )

    def forward(self, x, y):
        down1 = self.down1(x)
        down2 = self.down2(down1)

        down3 = self.down3(y)
        down4 = self.down4(down3)
        out = self.mid(torch.cat([down2, down4], dim=1))
        if out.shape != down4.shape:
            out = F.interpolate(out, down4.size()[-2:])
        out = self.up1(torch.cat([out, down4], dim=1))
        if out.shape != down3.shape:
            out = F.interpolate(out, down3.size()[-2:])
        out = self.up2(torch.cat([out, down3], dim=1))
        return out

class JointModel(nn.Module):
    def __init__(self):
        super(JointModel, self).__init__()
        self.stage1 = InpaintNet()
        self.kernel_sizes = [(1, 7), (7, 1), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5),
                             (5, 5), (5, 5), (5, 5)]
        self.dilations = [(1, 1), (1, 1), (1, 1), (2, 1), (4, 1), (8, 1), (16, 1), (32, 1), (1, 1), (2, 2), (4, 4),
                          (8, 8), (16, 16), (32, 32)]
        self.stage2 = ContextAggNet(self.kernel_sizes, self.dilations)

    def forward(self, x, n):
        n_pred = self.stage1(n, x)
        out = self.stage2(x, n_pred)
        return n_pred, out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size:tuple, dilation:tuple,
                 stride=1,
                 norm_fn='bn',
                 act='relu'):
        super(ConvBlock, self).__init__()
        pad = ((kernel_size[0] - 1) // 2 * dilation[0], (kernel_size[1] - 1) // 2 * dilation[1])
        block = []
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, dilation, bias=norm_fn is None))
        if norm_fn == 'bn':
            block.append(nn.BatchNorm2d(out_channels))
        if act == 'relu':
            block.append(nn.ReLU())
        elif act == 'prelu':
            block.append(nn.PReLU())
        elif act == 'lrelu':
            block.append(nn.LeakyReLU())
        elif act == 'tanh':
            block.append(nn.Tanh())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class ContextAggNet(nn.Module):
    def __init__(self,
                 kernel_sizes,
                 dilations,
                 freq_bins=256,
                 nf=96):
        super(ContextAggNet, self).__init__()
        self.encoder_x = self.make_enc(kernel_sizes, dilations, nf)
        self.encoder_n = self.make_enc(kernel_sizes, dilations, nf // 2, outf=4)

        self.lstm = nn.LSTM(input_size=8*freq_bins + 4*freq_bins, hidden_size=200, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(400, 600),
                                nn.ReLU(True),
                                nn.Linear(600, 600),
                                nn.ReLU(True),
                                nn.Linear(600, freq_bins * 2),
                                nn.Sigmoid())

    def make_enc(self, kernel_sizes, dilations, nf=96, outf=8):
        encoder_x = []
        for i in range(len(kernel_sizes)):
            if i == 0:
                encoder_x.append(ConvBlock(2, nf, kernel_sizes[i], dilations[i]))
            else:
                encoder_x.append(ConvBlock(nf, nf, kernel_sizes[i], dilations[i]))
        encoder_x.append(ConvBlock(nf, outf, (1, 1), (1, 1)))
        return nn.Sequential(*encoder_x)

    def forward(self, x, n):
        f_x = self.encoder_x(x)
        f_x = f_x.view(f_x.size(0), -1, f_x.size(3)).permute(2, 0, 1)
        f_n = self.encoder_n(n)
        f_n = f_n.view(f_n.size(0), -1, f_n.size(3)).permute(2, 0, 1)
        # if self.training is True:
        self.lstm.flatten_parameters()
        f_x, _ = self.lstm(torch.cat([f_x, f_n], dim=2))
        f_x = f_x.permute(1, 0, 2)
        f_x = self.fc(f_x)
        out = f_x.permute(0, 2, 1).view(f_x.size(0), 2, -1, f_x.size(1))
        # f_n = self.encoder_n(n)
        return out



class DeepSpeech(pl.LightningModule):
    def __init__(self,
                 wandb,
                 #net,
                 #net_two,
                 power,
                 future_amt: float,
                 future: bool,
                 residual: bool,
                 batchnorm:  bool,
                 waveform: bool,
                 capped: bool,
                 inputreal: bool,
                 firstlayer: bool,
                 labels: List,
                 model_cfg: DictConfig,
                 precision: int,
                 optim_cfg: DictConfig,
                 spect_cfg: DictConfig
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.wandb = wandb
        self.waveform = waveform
        self.future = future
        self.capped = capped
        self.model_cfg = model_cfg
        self.power = power
        self.firstlayer = firstlayer
        self.precision = precision
        self.inputreal = inputreal #ignore
        self.optim_cfg = optim_cfg
        self.mag_noise = model_cfg.mag_noise
        self.spect_cfg = spect_cfg
        self.bidirectional = True
        self.residual=residual
        self.batchnorm = batchnorm

        #self.net = net
        #self.net_two = net_two

        self.labels = labels
        num_classes = len(self.labels)
        if self.batchnorm:
            if self.residual:
                self.halfsec = HalfSecNetResidual()
            else:
                if self.waveform:
                    self.halfsec = HalfSecNetWav()
                else:
                    self.halfsec = HalfSecNet()

        else:
            if self.residual:
                self.halfsec = HalfSecNetResidualNoBN()
            else:
                self.halfsec = HalfSecNetNoBN()

        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        ))
        self.tanh = torch.nn.Tanh()
        self.criterion = CTCLoss(blank=self.labels.index('_'), reduction='sum', zero_infinity=True)
        #self.criterion = torch.nn.MSELoss() #CosineSimilarity()  # CTCLoss(blank=self.labels.index('_'), reduction='sum', zero_infinity=True)
        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = int(math.floor((self.spect_cfg.sample_rate * self.spect_cfg.window_size) / 2) + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size *= 32

        self.rnns = nn.Sequential(
            BatchRNN(
                input_size=rnn_input_size,
                hidden_size=self.model_cfg.hidden_size,
                rnn_type=nn.LSTM, #self.model_cfg.rnn_type.value
                bidirectional=self.bidirectional,
                batch_norm=False
            ),
            *(
                BatchRNN(
                    input_size=self.model_cfg.hidden_size,
                    hidden_size=self.model_cfg.hidden_size,
                    rnn_type=nn.LSTM, #self.model_cfg.rnn_type.value
                    bidirectional=self.bidirectional
                ) for x in range(self.model_cfg.hidden_layers - 1)
            )
        )

        self.mseloss = torch.nn.MSELoss()
        self.future_amt = future_amt
        self.lookahead = nn.Sequential(
            # consider adding batch norm?
            Lookahead(self.model_cfg.hidden_size, context=self.model_cfg.lookahead_context),
            nn.Hardtanh(0, 20, inplace=True)
        ) if not self.bidirectional else None

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(self.model_cfg.hidden_size, track_running_stats=False),
            nn.Linear(self.model_cfg.hidden_size, num_classes, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_softmax = InferenceBatchSoftmax()
        self.evaluation_decoder = GreedyDecoder(self.labels)  # Decoder used for validation
        self.wer = WordErrorRate(
            decoder=self.evaluation_decoder,
            target_decoder=self.evaluation_decoder
        )
        self.cer = CharErrorRate(
            decoder=self.evaluation_decoder,
            target_decoder=self.evaluation_decoder
        )
        self.n_fft_second = 510
        self.hop_length_second = 158
        self.win_length_second = 400
        self.frames_to_audiosample_ratio = float(16000) / 30


    def run_through_first_conv(self, xdelta, output_lengths):
        xdelta = torch.log1p(xdelta)
        mean = torch.mean(xdelta)
        std = torch.std(xdelta)
        xdelta = xdelta - mean
        xdelta = xdelta / std
        # pdb.set_trace()
        f_xdelta, _ = self.conv(xdelta, output_lengths)
        return f_xdelta

    def stftistft(self, actual_noise):
        if self.capped and self.inputreal:
            all_noise_w = self.power * torch.tanh(
                torch.istft(actual_noise.permute((0, 2, 3, 1)), n_fft=320, win_length=320,
                            hop_length=int(16000 * 0.01)))
            actual_noise_scaled = torch.stft(all_noise_w, n_fft=320, win_length=320,
                                             hop_length=int(16000 * 0.01)).permute(
                (0, 3, 1, 2))
            mag_noise = torch.sqrt(actual_noise_scaled[:, 0, :, :] ** 2 + actual_noise_scaled[:, 1, :, :] ** 2)


        elif not self.capped and self.inputreal:
            all_noise_w = torch.tanh(torch.istft(actual_noise.permute((0, 2, 3, 1)), n_fft=320, win_length=320,
                                                 hop_length=int(16000 * 0.01)))
            actual_noise_scaled = torch.stft(all_noise_w, n_fft=320, win_length=320,
                                             hop_length=int(16000 * 0.01)).permute(
                (0, 3, 1, 2))
            mag_noise = torch.sqrt(actual_noise_scaled[:, 0, :, :] ** 2 + actual_noise_scaled[:, 1, :, :] ** 2)

        else:
            all_noise_w = torch.istft(actual_noise.permute((0, 2, 3, 1)), n_fft=320, win_length=320,
                                      hop_length=int(16000 * 0.01))
            actual_noise_scaled = torch.stft(all_noise_w, n_fft=320, win_length=320,
                                             hop_length=int(16000 * 0.01)).permute(
                (0, 3, 1, 2))
            mag_noise = torch.sqrt(actual_noise[:, 0, :, :] ** 2 + actual_noise[:, 1, :, :] ** 2)
        return all_noise_w, actual_noise_scaled, mag_noise

    def pad_w_zeros_stft(self, input, all_noise):
        if not self.future:
            zeros = torch.zeros(input.shape[0], input.shape[1], input.shape[2],
                                input.shape[3] - all_noise.shape[3] - 204).cuda()
            actual_noise = torch.cat([torch.zeros(input.shape[0], 2, 161, 204).cuda(), all_noise, zeros],
                                     axis=3) + 1e-9
        else:
            zeros = torch.zeros(input.shape[0], input.shape[1], input.shape[2],
                                input.shape[3] - all_noise.shape[3] - (204 + self.future_amt)).cuda()
            actual_noise = torch.cat(
                [torch.zeros(input.shape[0], 2, 161, (204 + self.future_amt)).cuda(), all_noise, zeros],
                axis=3) + 1e-9

        return actual_noise

    def run_through_full_network(self, mag_input, mag_noise, output_lengths):
        x = torch.unsqueeze(mag_input, dim=1) + torch.unsqueeze(mag_noise, dim=1)
        x = torch.log1p(x)
        mean = torch.mean(x)
        std = torch.std(x)
        x = x - mean
        x = x / std
        x, _ = self.conv(x, output_lengths)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH
        for rnn in self.rnns:
            x = rnn(x, output_lengths)

        if not self.bidirectional:  # no need for lookahead layer in bidirectional
            x = self.lookahead(x)

        x = self.fc(x)
        x = x.transpose(0, 1)
        # identity in training mode, softmax in eval mode
        x = self.inference_softmax(x)

        return x

    def firstlayerpass(self, mag_input, mag_noise, output_lengths):
        """Computed feature of x + delta (noise) """
        xdelta = torch.unsqueeze(mag_input, dim=1) + torch.unsqueeze(mag_noise, dim=1)
        f_xdelta = self.run_through_first_conv(xdelta, output_lengths)

        """Computed feature of x """
        x = torch.unsqueeze(mag_input, dim=1)
        f_x = self.run_through_first_conv(x, output_lengths)

        """Computed feature of x + permuted delta (noise) """

        with torch.no_grad():

            f_xdeltaclone = f_xdelta.clone()
            f_xdeltaclone = f_xdeltaclone.detach()
            sizes = f_xdeltaclone.size()
            f_xdeltaclone = f_xdeltaclone.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
            f_xdeltaclone = f_xdeltaclone.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH
            for rnn in self.rnns:
                f_xdeltaclone = rnn(f_xdeltaclone, output_lengths)

            if not self.bidirectional:  # no need for lookahead layer in bidirectional
                f_xdeltaclone = self.lookahead(f_xdeltaclone)

            f_xdeltaclone = self.fc(f_xdeltaclone)
            f_xdeltaclone = f_xdeltaclone.transpose(0, 1)
            # identity in training mode, softmax in eval mode
            f_xdeltaclone = self.inference_softmax(f_xdeltaclone)


        return f_x, f_xdelta, f_xdeltaclone

    def batch_fast_icRM_sigmoid(self, Y, crm, a=0.1, b=0):
        """
        :param Y: (B, 2, F, T)
        :param crm: (B, 2, F, T)
        :param a:
        :param b:
        :return:
        """
        M = 1. / a * (torch.log(crm / (1 - crm + 1e-8) + 1e-10) + b)
        r = M[:, 0, :, :] * Y[:, 0, :, :] - M[:, 1, :, :] * Y[:, 1, :, :]
        i = M[:, 0, :, :] * Y[:, 1, :, :] + M[:, 1, :, :] * Y[:, 0, :, :]
        rec = torch.stack([r, i], dim=1)
        return rec


    def forward(self, input, lengths, scalar):

        lengths = lengths.cpu().int()
        output_lengths = self.get_seq_lens(lengths)
        mag_input = torch.sqrt(input[:, 0, :, :] ** 2 + input[:, 1, :, :] ** 2)
        list_of_noise = []

        if int((input.shape[3]- (204+self.future_amt))/ 51) > 0:
            for i in range(int((input.shape[3]- (204+self.future_amt))/ 51)):

                out = self.halfsec(input[:, :, :,i * 51:204 + i * 51])

                list_of_noise.append(out)

        if len(list_of_noise) > 0:
            noise=torch.cat(list_of_noise, axis=2)[:,0,:]
            all_noise_w = self.power*torch.unsqueeze(scalar.cuda(),dim=1)*noise
            all_noise = torch.stft(all_noise_w, n_fft=320, win_length=320, hop_length=int(16000 * 0.01))
            all_noise = all_noise.permute((0, 3, 1,
                                           2))
            minus = lengths[0] - all_noise.shape[3]

            actual_noise = self.pad_w_zeros_stft(input, all_noise)
            mask=input!=0
            actual_noise = actual_noise*mask + 0.0000001
            mag_noise = torch.sqrt(actual_noise[:, 0, :, :] ** 2 + actual_noise[:, 1, :, :] ** 2)


            x = self.run_through_full_network(mag_input, mag_noise, output_lengths)

            return x, output_lengths, actual_noise, actual_noise, mag_noise, mag_input, torch.max(torch.abs(all_noise_w))

        else:
            return None


    def training_step(self, batch, batch_idx):

        inputs, targets, mag_noises, input_percentages, target_sizes, scalar = batch

        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

        output = self.forward(inputs,input_sizes,scalar)

        if output is not None:
            x, output_lengths, actual_noise, actual_noise, mag_noise, mag_input, max_y= output
            out = x.transpose(0, 1)  # TxNxH
            out = out.log_softmax(-1)
            loss_ctc = self.criterion(out, targets, output_lengths,target_sizes)

            with torch.no_grad():
                decoded_output, _ = self.evaluation_decoder.decode(x, output_lengths)
                d_target = [self.labels[targets[i].item()] for i in range(len(targets))]
                decoded_target = ""
                decoded_target = decoded_target.join(d_target[:target_sizes[0]])
                lev_dist = self.wer_calc(decoded_target, decoded_output[0][0])
                nwords = len(decoded_target.split())
                nchars = len(decoded_target.replace(' ', ''))
                cer_dist = self.cer_calc(decoded_target, decoded_output[0][0])


        else:
            pdb.set_trace()
            return {"loss":torch.tensor([0.0]), "lev_dist": float(0), "lev_dist_shifted": float(0), "nwords": float(0), "nchars": float(0), "cer_dist": float(0)}

        if self.wandb:
            self.log('ctc_loss',loss_ctc,on_step=True, sync_dist=True)
            self.log('wer_train',lev_dist / float(nwords),on_step=True, sync_dist=True)
            self.log('max_abs',max_y.cuda(),on_step=True, sync_dist=True)


        return {"loss":-loss_ctc, "lev_dist": float(lev_dist), "nwords": float(nwords), "nchars": float(nchars), "cer_dist": float(cer_dist)}

    def training_epoch_end(self, train_step_outputs):
        if self.waveform:
            both = [[dict["loss"] for dict in train_step_outputs],
                   [dict["lev_dist"] for dict in train_step_outputs],
                    [dict["nwords"] for dict in train_step_outputs]]
            loss, lev_dist, n_words= both[0], both[1], both[2]
        else:
            both = [[dict["loss"] for dict in train_step_outputs],
                    [dict["lev_dist"] for dict in train_step_outputs],
                    [dict["nwords"] for dict in train_step_outputs]]
            loss, lev_dist, n_words = both[0], both[1], both[2]

        self.logger.experiment.log({"epoch_loss_train":torch.mean(torch.tensor(loss)).cuda()})
        self.logger.experiment.log({"wer_train_epoch":torch.mean(torch.tensor(lev_dist)).cuda() / torch.mean(torch.tensor(n_words)).cuda()})


    def validation_step(self, batch, batch_idx):
        inputs, targets, mag_noises, input_percentages, target_sizes, scalar = batch

        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()


        output = self.forward(inputs, input_sizes,scalar)
        if output is not None:
            x, output_lengths, actual_noise, actual_noise_scaled, mag_noise, mag_input, max_y = output
            out = x.transpose(0, 1)  # TxNxH
            out = out.log_softmax(-1)
            loss_ctc = self.criterion(out, targets, output_lengths, target_sizes)


            with torch.no_grad():
                decoded_output, _ = self.evaluation_decoder.decode(x, output_lengths)
                d_target = [self.labels[targets[i].item()] for i in range(len(targets))]
                decoded_target = ""
                decoded_target = decoded_target.join(d_target[:target_sizes[0]])
                lev_dist = self.wer_calc(decoded_target, decoded_output[0][0])
                nwords = len(decoded_target.split())
                nchars = len(decoded_target.replace(' ', ''))
                cer_dist = self.cer_calc(decoded_target, decoded_output[0][0])

        else:
            return {"loss":torch.tensor([0.0]), "lev_dist": float(0), "nwords": float(0), "nchars": float(0), "cer_dist": float(0)}

        if self.wandb:

            self.logger.experiment.log({'val_ctc_loss': loss_ctc,
                                        "global_step": self.global_step})
            self.logger.experiment.log({'wer_val': lev_dist / float(nwords),
                                        "global_step": self.global_step})

            self.log('max_abs_val',max_y.cuda(),on_step=True, sync_dist=True)

        return {"loss": -loss_ctc, "lev_dist": float(lev_dist),
                    "nwords": float(nwords), "nchars": float(nchars), "cer_dist": float(cer_dist)}


    def validation_epoch_end(self, val_step_outputs):
        if self.waveform:
            both = [[dict["loss"] for dict in val_step_outputs],
                    [dict["lev_dist"] for dict in val_step_outputs],
                    [dict["nwords"] for dict in val_step_outputs]]
            loss, lev_dist, n_words  = both[0], both[1], both[2]
        else:
            both = [[dict["loss"] for dict in val_step_outputs],
                    [dict["lev_dist"] for dict in val_step_outputs],
                    [dict["nwords"] for dict in val_step_outputs]]
            loss, lev_dist, n_words = both[0], both[1], both[2]
        self.logger.experiment.log({"epoch_loss_val":torch.mean(torch.tensor(loss)).cuda()})
        self.logger.experiment.log({"wer_valepoch":torch.mean(torch.tensor(lev_dist)).cuda() / torch.mean(torch.tensor(n_words)).cuda()})


    def cer_calc(self, s1, s2):
        """
        Computes the Character Error Rate, defined as the edit distance.

        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
        s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
        return Lev.distance(s1, s2)

    def wer_calc(self, s1, s2):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """

        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        return Lev.distance(''.join(w1), ''.join(w2))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.halfsec.parameters(),
            lr=self.optim_cfg.learning_rate,
            eps=self.optim_cfg.eps,
            weight_decay=self.optim_cfg.weight_decay)
        """optimizer = torch.optim.SGD(
            params=self.halfsec.parameters(),
            lr=self.optim_cfg.learning_rate)"""

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=self.optim_cfg.learning_anneal
        )
        return [optimizer], [scheduler]

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) // m.stride[1] + 1)
        return seq_len.int()



