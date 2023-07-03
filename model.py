# -*- coding: utf-8 -*-
# @Time : 2022/2/28 12:40
# @Author : jklujklu
# @Email : jklujklu@126.com
# @File : models.py
# @Software: PyCharm
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # (3,30,30) -> (32,28,28)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=1, padding=1)
        # (32,28,28) -> (32,14,14)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # (32,14,14) -> (64,12,12)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=1, padding=1)
        # (64,12,12) -> (64,6,6)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # (64,6,6) -> (64,4,4)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=1, padding=1)
        self.fl = nn.Flatten()
        self.fc1 = nn.Linear(4 * 4 * 64, 64)
        self.fc2 = nn.Linear(64, 43)

    def forward(self, inputs):
        tensor = inputs.view(-1, 3, 30, 30)
        tensor = F.relu(self.conv1(tensor))
        # print(tensor.shape)
        tensor = self.pool1(tensor)
        # print(tensor.shape)
        tensor = F.dropout(tensor, p=0.25)
        # print(tensor.shape)
        tensor = F.relu(self.conv2(tensor))
        # print(tensor.shape)
        tensor = self.pool2(tensor)
        # print(tensor.shape)
        tensor = F.dropout(tensor, p=0.25)
        tensor = F.relu(self.conv3(tensor))
        # tensor = self.fl(tensor
        #                  )
        # print(tensor.shape)
        tensor = tensor.view(-1, 4 * 4 * 64)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor
