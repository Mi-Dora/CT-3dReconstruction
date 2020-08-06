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
import time

default_weight_path = '../weights'
default_log_path = '../log'


def train(data_dir, epochs, batch_size, check_point, weight_file=''):
    # prepare
    os.makedirs(default_weight_path, exist_ok=True)
    os.makedirs(default_log_path, exist_ok=True)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # initialization
    dataloader = DataLoader(
        LIDC(data_dir, 128),
        batch_size=batch_size,
        shuffle=True
    )
    model = SurfaceNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00025, momentum=0.9, nesterov=True)
    # loss_func = CBCELoss(0.5)
    loss_func = torch.nn.MSELoss()

    now = int(round(time.time() * 1000))
    now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(now / 1000))
    epoch_log_name = '../log/batch_log_{}.txt'.format(now)
    epoch_trained = 0

    # load weight
    if weight_file == 'latest':
        for root, _, files in os.walk(default_weight_path):
            model.load_state_dict(torch.load(os.path.join(root, files[-1])))
            epoch_trained = int(files[-1].split('.')[0])
        for root, _, files in os.walk(default_log_path):
            epoch_log_name = os.path.join(root, files[-1])
    elif weight_file != '':
        model.load_state_dict(torch.load(weight_file))
        epoch_trained = int(weight_file.split('/')[-1].split('.')[0])
        for root, _, files in os.walk(default_log_path):
            epoch_log_name = os.path.join(root, files[-1])

        # begin training
    for epoch in range(epochs):
        tt_loss = 0
        for i, (ct, xray) in enumerate(dataloader):
            ct = ct.to(device)
            xray = xray.to(device)
            optimizer.zero_grad()
            # forward
            output = model(xray)
            # compute loss
            loss = loss_func(output, ct)
            tt_loss += loss.item()
            # backward
            loss.backward()
            optimizer.step()
            print("[Epoch %d/%d] [Batch %d/%d] [loss: %f]"
                  % (epoch + epoch_trained, epochs, i, len(dataloader), loss.item())
                  )

            with open('../log/batch_log_{}.txt'.format(now), 'a') as b_logger:
                b_logger.writelines("%d\t%d\t%f\n" % (epoch + epoch_trained, i, loss.item()))
        with open(epoch_log_name, 'a') as e_logger:
            e_logger.writelines("%d\t%f\n" % (epoch + epoch_trained, tt_loss))
        print('====> Epoch: {} Average loss: {:.6f}'.format(
            epoch + epoch_trained, tt_loss / len(dataloader.dataset)))
        if (epoch + 1) % check_point == 0:
            torch.save(model.state_dict(), '{}/{:0>4d}.pkl'.format(default_weight_path, epoch + epoch_trained + 1))


if __name__ == '__main__':
    train(
        data_dir='G:/CT-LIDC/LIDC-HDF5-256/',
        epochs=700,
        batch_size=3,
        check_point=1,
        weight_file='../weights/0446.pkl'
    )



