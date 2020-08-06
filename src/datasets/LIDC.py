# -*- coding: utf-8 -*-
"""
    Created on Tuesday, Aug 4 2020

    Author          ：Yu Du
    Email           : yuduseu@gmail.com
    Last edit date  : Wednesday, Aug 6 2020

Southeast University, College of Automation, 211189 Nanjing China
"""

from torch.utils import data
import torch
import h5py
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from math import *
from scipy.ndimage import zoom


class LIDC(data.Dataset):
    """
    Load custom dataset by giving the data folder path
    """

    def __init__(self, dir, size):
        """
        :param dim: (int) The dimension of the fake data, expected to be 1 or 2
        :param length: (int) length of the fake dataset
        """
        super(LIDC, self).__init__()

        self.dir = dir
        self.filename = 'ct_xray_data.h5'
        self.size = size
        self.save_image = False
        for root, dirs, _ in os.walk(self.dir):
            self.data_folders = dirs[:30]
            break

    def __getitem__(self, idx):
        """
        return: (tensor) data send to network
        obj[idx] == obj.__getitem__(idx)
        """
        data_folder = self.data_folders[idx]
        path = os.path.join(self.dir, data_folder, self.filename)
        data = h5py.File(path, 'r')
        ct = data['ct'][()]
        h, w, d = ct.shape
        assert h == w == d
        minVal = ct.min()
        maxVal = ct.max()
        ct = (ct - minVal) / (maxVal - minVal)
        ct_proj1 = (ct.sum(axis=0) * 2).astype('uint8')
        ct_proj2 = (ct.sum(axis=1) * 2).astype('uint8')
        ct_proj3 = (ct.sum(axis=2) * 2).astype('uint8')
        if self.save_image:
            cv2.imwrite('../../images/ct1_'+str(idx)+'.jpg', ct_proj1)
            cv2.imwrite('../../images/ct2_'+str(idx)+'.jpg', ct_proj2)
            cv2.imwrite('../../images/ct3_'+str(idx)+'.jpg', ct_proj3)
        ct = zoom(ct, self.size / h)
        xray1 = data['xray1'][()]
        xray2 = data['xray2'][()]
        if self.save_image:
            cv2.imwrite('../../images/xray1_'+str(idx)+'.jpg', xray1)
            cv2.imwrite('../../images/xray2_'+str(idx)+'.jpg', xray2)
        h, w = xray1.shape
        assert h == w
        xray1 = zoom(xray1, self.size / h)
        xray2 = zoom(xray2, self.size / h)
        merged = self.merge(xray1, xray2)
        ct = ct[np.newaxis, :, :, :]
        merged = merged[np.newaxis, :, :, :]
        ct = torch.FloatTensor(ct)
        merged = torch.FloatTensor(merged)
        return ct, merged

    def __len__(self):
        return len(self.data_folders)

    def merge(self, xray1, xray2):
        h, w = xray1.shape
        assert h == w == self.size
        xray1 = (xray1/2)[:, np.newaxis, :]
        xray2 = (xray2/2)[:, :, np.newaxis]
        for _ in range(int(log(h, 2))):
            xray1 = np.concatenate((xray1, xray1), axis=1)
            xray2 = np.concatenate((xray2, xray2), axis=2)
        # xray1 = xray1.astype('uint8')
        # xray2 = xray2.astype('uint8')
        merged = (xray1 + xray2)
        minVal = merged.min()
        maxVal = merged.max()
        merged = (merged - minVal) / (maxVal - minVal)
        # for i in range(256):
        #     img = merged[:, :, i]
        #     cv2.imshow('piece', img)
        #     cv2.waitKey(200)
        return merged


if __name__ == '__main__':
    dir = 'G:/CT-LIDC/LIDC-HDF5-256/'
    dataloader = data.DataLoader(
        LIDC(dir, 128),
        batch_size=1,
        shuffle=True
    )
    for i, (ct, xray) in enumerate(dataloader):
        pass

    # data = h5py.File(dir + 'LIDC-IDRI-0001.20000101.3000566.1/ct_xray_data.h5', 'r')
    # xray1 = data['xray1'][()]
    # ct = data['ct'][()]
    # minVal = ct.min()
    # maxVal = ct.max()
    # ct = ((ct - minVal) / (maxVal - minVal) * 256).astype('uint8')
    #
    # lapalian_demo(xray1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # for i in range(256):
    #     img = ct[:, i, :]
    #     cv2.imshow('piece', ct[:, i, :])
    #     cv2.waitKey(200)

