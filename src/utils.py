# -*- coding: utf-8 -*-
"""
    Created on Tuesday, Aug 4 2020

    Author          ：Yu Du
    Email           : yuduseu@gmail.com
    Last edit date  : Wednesday, Aug 6 2020

Southeast University, College of Automation, 211189 Nanjing China
"""

import numpy as np
import scipy.io as scio

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def plot3D(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    z = data[:, 0]
    y = data[:, 1]
    x = data[:, 2]
    d = data[:, 3]
    ax.scatter(x, y, z, c='b', marker='o', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def save_mat(key, data):
    dataNew = key + '.mat'
    scio.savemat(dataNew, {key: data})


def change2vector(matrix):
    h, w, d = matrix.shape
    new_mat = np.zeros((h * w * d, 4))
    matrix = matrix.reshape(1, -1)
    new_mat[:, 3] = matrix
    arr = np.arange(1, 129)
    for _ in range(14):
        arr = np.concatenate((arr, arr), axis=0)
    arr.reshape(1, -1)
    new_mat[:, 2] = arr
    arr = np.ones((1, 128))
    for y in range(w):
        new_mat[y * 128:(y + 1) * 128, 1] = arr
        arr += 1
    arr = np.ones((1, 128 * 128))
    length = 128 * 128
    for z in range(h):
        new_mat[z * length:(z + 1) * length, 0] = arr
        arr += 1
    # new_mat = (new_mat*256).astype(np.int)
    return new_mat


def enhance_contrast(data):
    minVal = data.min()
    maxVal = data.max()
    data = (data - minVal) / (maxVal - minVal)
    return data


if __name__ == '__main__':
    # # random input data
    output = np.zeros((128, 128, 128))
    output = change2vector(output)
    save_mat('test1', output)
    # gt = torch.tensor(torch.randint(0, 2, (8, 1, 100, 100)), dtype=torch.float, requires_grad=True)
    # loss_func = CBCELoss(0.4)
    # loss = loss_func(output, gt)
    # print(loss)
    # pass
