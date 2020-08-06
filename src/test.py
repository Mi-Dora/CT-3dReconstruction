# -*- coding: utf-8 -*-
"""
    Created on Tuesday, Aug 4 2020

    Author          ：Yu Du
    Email           : yuduseu@gmail.com
    Last edit date  : Wednesday, Aug 6 2020

Southeast University, College of Automation, 211189 Nanjing China
"""

import os
import torch
from torch.utils.data import DataLoader
from src.models import SurfaceNet
from src.datasets import LIDC
from src.utils import save_mat, enhance_contrast
import numpy as np


default_weight_path = '../weights'
default_image_path = '../images'
default_mat_path = '../mats'


def test(data_dir, batch_size=1, weight_file=''):
    # prepare
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # initialization
    dataloader = DataLoader(
        LIDC(data_dir, 128),
        batch_size=batch_size,
        shuffle=False
    )
    model = SurfaceNet().to(device)
    for param in model.parameters():
        param.requires_grad = False

    # load weight
    if weight_file == 'latest':
        for root, _, files in os.walk(default_weight_path):
            model.load_state_dict(torch.load(os.path.join(root, files[-1])))
    elif weight_file != '':
        model.load_state_dict(torch.load(weight_file))

    for i, (ct, xray) in enumerate(dataloader):
        ct = ct.to(device)
        xray = xray.to(device)

        # forward
        output = model(xray)
        output = output.squeeze()
        # output = enhance_contrast(output.cpu().numpy())
        output = output.cpu().numpy().astype(np.float)

        ct = ct.squeeze()
        ct = ct.cpu().numpy()
        save_mat('output' + str(i), output)
        save_mat('ct' + str(i), ct)


        # xray = xray.squeeze()
        # xray = xray.cpu().numpy()
        # xray_proj2 = (enhance_contrast(xray.sum(axis=1)) * 255).astype('uint8')
        # xray_proj1 = (enhance_contrast(xray.sum(axis=0)) * 255).astype('uint8')
        # xray_proj3 = (enhance_contrast(xray.sum(axis=2)) * 255).astype('uint8')
        # cv2.imwrite('../images/input1_' + str(i) + '.jpg', xray_proj1)
        # cv2.imwrite('../images/input2_' + str(i) + '.jpg', xray_proj2)
        # cv2.imwrite('../images/input3_' + str(i) + '.jpg', xray_proj3)

        # save_mat('../mats/output_vec_' + str(i), change2vector(output))
        # save_mat('../mats/ct_vec_' + str(i), change2vector(ct))
        # npoutput = np.concatenate((output, ct), axis=2)
        # npoutput_x = np.concatenate((output, ct), axis=1)
        # npoutput = (npoutput * 256).astype('uint8')
        # npoutput_x = (npoutput_x * 256).astype('uint8')
        #
        # output_proj2 = (enhance_contrast(output.sum(axis=1))*255).astype('uint8')
        # output_proj1 = (enhance_contrast(output.sum(axis=0))*255).astype('uint8')
        # output_proj3 = (enhance_contrast(output.sum(axis=2))*255).astype('uint8')
        #
        # cv2.imwrite('../images/output1_'+str(i)+'.jpg', output_proj1)
        # cv2.imwrite('../images/output2_'+str(i)+'.jpg', output_proj2)
        # cv2.imwrite('../images/output3_'+str(i)+'.jpg', output_proj3)

        # for i in range(256):
        #     cv2.imshow('piece', npoutput[:, i, :])
        #     cv2.waitKey(200)
        # encoder = cv2.VideoWriter_fourcc(*'mp4v')
        # writer_x = cv2.VideoWriter()
        # video_save_path = '../videos/x_{}.mp4'.format(i)
        # writer_x.open(video_save_path, encoder, fps=10, frameSize=(256, 128), isColor=False)
        print(i)
        # for j in range(128):
        #     frame = npoutput_x[:, :, j]
        #     writer_x.write(frame)
        # writer_x.release()
        #
        # writer_y = cv2.VideoWriter()
        # video_save_path = '../videos/y_{}.mp4'.format(i)
        # writer_y.open(video_save_path, encoder, fps=10, frameSize=(256, 128), isColor=False)
        # for j in range(128):
        #     frame = npoutput[:, j, :]
        #     writer_y.write(frame)
        # writer_y.release()
        #
        # writer_z = cv2.VideoWriter()
        # video_save_path = '../videos/z_{}.mp4'.format(i)
        # writer_z.open(video_save_path, encoder, fps=10, frameSize=(256, 128), isColor=False)
        # for j in range(128):
        #     frame = npoutput[j, :, :]
        #     writer_z.write(frame)
        # writer_z.release()


if __name__ == '__main__':
    os.makedirs(default_mat_path, exist_ok=True)
    os.makedirs(default_image_path, exist_ok=True)
    os.makedirs('../videos', exist_ok=True)
    test(data_dir='G:/CT-LIDC/LIDC-HDF5-256/', batch_size=1, weight_file='../weights/0707.pkl')


