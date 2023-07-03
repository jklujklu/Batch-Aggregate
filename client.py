# -*- coding: utf-8 -*-
# @Time : 2022/9/26 21:46
# @Author : jklujklu
# @Email : jklujklu@126.com
# @File : client.py
# @Software: PyCharm
import time
from functools import reduce

import rsa
import torch
from charm.core.math.pairing import ZR
from charm.toolbox.pairinggroup import PairingGroup
from torch import nn, optim
import numpy as np
import torch.nn.functional as F
from loguru import logger

from algorithm.sign import ShortSig
from model import CNN


class BatchClient(object):
    def __init__(self, cid, train_dl, sign_params, encrypt_param):
        self.net = None
        self.opti = None
        self.loss_func = None
        self.dev = None
        self.id = cid
        self.train_dl = train_dl
        self.group = PairingGroup('SS512')
        self.sign_pub_key, self.sign_priv_key = sign_params

        self.encrypt_pub_key = encrypt_param

    def __sign(self, msg):
        sigma, timestamp = ShortSig.sign(self.group, self.sign_pub_key, self.sign_priv_key, msg)
        return sigma, msg, timestamp

    def __cramer_encrypt(self, msg):
        pk1, pk2, pk3, k1, k2 = self.encrypt_pub_key

        w_ = self.group.random(ZR)
        z1 = k1 * w_
        z2 = k2 * w_
        if type(msg) == int:
            eng = pk3 * w_ + msg
        elif type(msg) == np.ndarray:
            eng = int(pk3 * w_) + msg
        else:
            eng = ''
        h = self.group.hash((z1, z2, eng), ZR)
        z3 = pk1 * w_ + pk2 * w_ * h
        return z1, z2, eng, z3

    def local_train(self, params, lr, epoch):
        # Config
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        net = CNN()
        self.loss_func = F.cross_entropy
        self.opti = optim.SGD(net.parameters(), lr=lr)
        self.net = net.to(self.dev)

        # Load Global model
        par = net.state_dict().copy()
        for key, param in zip(par.keys(), params):
            par[key] = torch.from_numpy(param)
        net.load_state_dict(par, strict=True)
        self.net = net.to(self.dev)

        # Training
        for epoch in range(epoch):
            train_loss = 0
            batches = 0
            sum_accu = 0
            num = 0
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = net(data)
                loss = self.loss_func(preds, label)
                loss.backward()
                self.opti.step()
                self.opti.zero_grad()
                train_loss += loss.item()
                batches += 1
                _, preds = torch.max(preds.data, 1)

                sum_accu += (preds == label).float().mean()
                num += 1
            logger.debug('\tuser: {} | epoch: {} | Loss: {:.3f} | Acc: {:.3f}'.
                         format(self.id, epoch, train_loss / (batches + 1), sum_accu / num))

        # Encryption
        params = []
        par = self.net.state_dict().copy()
        for key in par.keys():
            _ = par[key].cpu().numpy()
            params.append(np.around((_ + 1.5) / 3 * 1024).astype(np.int))
        eng_msg = self.__cramer_encrypt(np.array(params))

        # Signature
        sign = self.__sign(eng_msg[2])

        return eng_msg, sign

    def local_val(self, test_dl):
        self.net.eval()
        sum_accu = 0
        num = 0
        for data, label in test_dl:
            data, label = data.to(self.dev), label.to(self.dev)
            preds = self.net(data)
            preds = torch.argmax(preds, dim=1)

            sum_accu += (preds == label).float().mean()
            num += 1
        logger.info('user:{} | local val acc: {:.3f}'.format(self.id, sum_accu / num))
