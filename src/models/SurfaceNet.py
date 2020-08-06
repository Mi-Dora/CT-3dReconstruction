# -*- coding: utf-8 -*-
"""
    Created on Tuesday, Aug 4 2020

    Author          ：Yu Du
    Email           : yuduseu@gmail.com
    Last edit date  : Wednesday, Aug 6 2020

Southeast University, College of Automation, 211189 Nanjing China
"""

import torch.nn as nn
import torch


class SurfaceNet(nn.Module):
    """
    SurfaceNet 3D accepts two CVCs as input
    Output the confidence of the voxel to be in the surface
    """
    def __init__(self):
        super(SurfaceNet, self).__init__()

        self.l1 = BnConvReLu3d(in_channels=1, out_channels=4, kernel_size=3, padding=1, max_pool=True)
        self.l2 = BnConvReLu3d(in_channels=4, out_channels=16, kernel_size=3, padding=1, max_pool=True)
        self.l3 = BnConvReLu3d(in_channels=16, out_channels=64, kernel_size=3, padding=1)
        self.l4 = BnConvReLu3d(in_channels=64, out_channels=256, kernel_size=3, padding=2, dilation=2)
        self.l5 = nn.Sequential(
            nn.BatchNorm3d(8),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(16)
        )
        self.s1 = BnUpConvSig3d(in_channels=4, out_channels=8, up_rate=2, kernel_size=1)
        self.s2 = BnUpConvSig3d(in_channels=16, out_channels=8, up_rate=4, kernel_size=1)
        self.s3 = BnUpConvSig3d(in_channels=64, out_channels=8, up_rate=4, kernel_size=1)
        self.s4 = BnUpConvSig3d(in_channels=256, out_channels=8, up_rate=4, kernel_size=1)
        self.output_layer = nn.Sequential(
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        :param cvc1: (tensor) size:(N, 3, s, s, s)
        :param cvc2: (tensor) size:(N, 3, s, s, s)
        :return: (tensor) size:(N, 1, s, s, s)
        """
        # cvc = torch.cat((cvc1, cvc2), dim=1)  # (N, C, H, W)
        lo1 = self.l1(input)
        lo2 = self.l2(lo1)
        lo3 = self.l3(lo2)
        lo4 = self.l4(lo3)
        so1 = self.s1(lo1)
        so2 = self.s2(lo2)
        so3 = self.s3(lo3)
        so4 = self.s4(lo4)
        sum_so = so1 + so2 + so3 + so4
        sum_so = self.l5(sum_so)
        return self.output_layer(sum_so)


class BnConvReLu3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, dilation=1, max_pool=False):
        super(BnConvReLu3d, self).__init__()
        self.max_pool = max_pool
        self.layer = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(inplace=False),
            nn.BatchNorm3d(out_channels)
        )
        if self.max_pool:
            self.MaxPooling = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):
        out = self.layer(x)
        if self.max_pool:
            out = self.MaxPooling(out)
        return out


class BnUpConvSig3d(nn.Module):
    def __init__(self, in_channels, out_channels, up_rate, kernel_size=1, padding=0):
        """
        :param in_channels: (int) number of input features
        :param out_channels: (int) number of output features
        :param up_rate: (int) output / input (side-length)
        :param kernel_size: (int or tuple)
        :param padding: (int)
        """
        super(BnUpConvSig3d, self).__init__()
        stride = up_rate
        self.layer = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding=up_rate - 1),
            nn.Sigmoid()
        )
        # output_padding: padding for the output to make the dimension meet the requirement,
        # which could be computed through formula in
        # https://pytorch.org/docs/master/generated/torch.nn.ConvTranspose3d.html#torch.nn.ConvTranspose3d

    def forward(self, x):
        # s_out = s_in * up_rate
        return self.layer(x)


if __name__ == '__main__':
    model = SurfaceNet()
    # random input data
    cvc1 = torch.rand(1, 1, 128, 128, 128)
    # cvc2 = torch.rand(1, 1, 128, 128, 128)
    output = model(cvc1)
    print(cvc1.shape)
    print(output.shape)


