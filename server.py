# -*- coding: utf-8 -*-
# @Time : 2022/9/26 21:46
# @Author : jklujklu
# @Email : jklujklu@126.com
# @File : server.py
# @Software: PyCharm
import os
import torch
from charm.core.math.pairing import ZR
from charm.toolbox.pairinggroup import PairingGroup
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from algorithm.sign import ShortSig
from dataset import GetDataSet
from model import CNN
from client import BatchClient
import torch.nn.functional as F

tic = None
bytes_length = 0


class BatchAggrServer:
    group = PairingGroup('SS512')

    def __init__(self):
        self.client_nums = 0  # 客户端数量
        self.aggregate = None  # 聚合模型
        self.global_model = []  # 全局模型
        self.lr = 0.01
        self.is_training = False
        self.sum_params = None
        self.nums = 4

        self.client_sids = []  # 客户端集合
        self.clients = {}

        # 短签名算法相关参数
        self.sign_pub, self.sign_master, self.sign_gamma = ShortSig.pubkey(self.group)
        # 客户端短签名私钥集
        self.sign_privs = {}
        # 加密参数
        self.crypt_secret, self.crypt_pub = None, None

    def __init_crammer(self):
        """
        初始化Cramer加密公私钥
        """
        sk1, sk2, sk3, sk4, sk5 = self.group.random(ZR), self.group.random(ZR), self.group.random(
            ZR), self.group.random(ZR), self.group.random(ZR)
        k1, k2 = self.group.random(ZR), self.group.random(ZR)
        pk1 = k1 * sk1 + k2 * sk2
        pk2 = k1 * sk3 + k2 * sk4
        pk3 = k1 * sk5
        self.crypt_pub = (pk1, pk2, pk3, k1, k2)
        self.crypt_secret = (sk1, sk2, sk3, sk4, sk5)

    def __init_global_model(self):
        """
        初始化全局模型以及机器学习相关配置
        """
        logger.info('init global model!')
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # # GTSRB
        net = CNN()
        self.loss_func = F.cross_entropy
        self.opti = optim.SGD(net.parameters(), lr=self.lr)
        self.net = net.to(self.dev)

        par = self.net.state_dict().copy()
        for key in par.keys():
            self.global_model.append(par[key].cpu().numpy())
        self.global_model = np.array(self.global_model)

    def __init_dataset(self):
        """
        初始化数据集，数据划分、类型转换
        """
        logger.info('init dataset!')
        # GTSRB
        x_train, x_test, y_train, y_test = GetDataSet('./GTSRB/Train').load_data()
        test_data = torch.tensor(x_test, dtype=torch.float32)
        test_label = torch.argmax(torch.tensor(y_test, dtype=torch.float32), dim=1)
        self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=64, shuffle=False,
                                           drop_last=True)

        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.argmax(torch.tensor(y_train, dtype=torch.float32), dim=1)
        self.train_data = x_train
        self.train_label = y_train

    def __init_clients(self):
        """
        初始化客户端，为每个客户端分配训练数据集及公私钥等
        """
        logger.info('init clients!')
        max_num = len(self.train_data)
        self.num_interval = int(max_num / self.nums)
        # self.num_interval = 500
        for i in range(self.nums):
            # 创建签名私钥
            priv = ShortSig.keygen(self.group, self.sign_pub, self.sign_gamma)
            self.sign_privs.update({f'client_{i}': priv})
            # 分配训练样本
            images = self.train_data[self.num_interval * i: self.num_interval * (i + 1)]
            labels = self.train_label[self.num_interval * i: self.num_interval * (i + 1)]
            train_dl = DataLoader(TensorDataset(images, labels), batch_size=64, shuffle=True, drop_last=True)
            # 创建训练客户端
            self.client_sids.append(f'client_{i}')
            self.clients.update(
                {f'client_{i}': BatchClient(f'client_{i}', train_dl, (self.sign_pub, priv), self.crypt_pub)})

    def start(self):
        logger.info('start training!')
        self.__init_crammer()
        self.__init_global_model()
        self.__init_dataset()
        self.__init_clients()

        for i in range(10):
            eng_msgs = []
            signs = []
            for client in self.clients.keys():
                en_msg, sign = self.clients[client].local_train(self.global_model, 0.01, 5)
                self.clients[client].local_val(self.test_data_loader)
                eng_msgs.append(en_msg)
                signs.append(sign)

            # Batch signatures verify
            rs = ShortSig.batch_verify(self.group, self.sign_pub, signs)
            if not rs:
                logger.info('Batch signatures verify failed, skip this round!')
                continue

            # Decrypt
            sum_parameters = 0
            for eng_msg in eng_msgs:
                z1, z2, eng, z3 = eng_msg
                sk1, sk2, sk3, sk4, sk5 = self.crypt_secret
                alpha = self.group.hash((z1, z2, eng), ZR)
                z3_ = z1 * (sk1 + alpha * sk3) + z2 * (sk2 + alpha * sk4)
                if z3 == z3_:
                    sum_parameters += eng - z1 * sk5

            # AVG
            index = 0
            for _ in sum_parameters:
                self.global_model[index] = sum_parameters[index].astype(np.int) / self.nums / 1024 * 3 - 1.5
                index += 1

            # Global model validation
            par = self.net.state_dict().copy()
            for key, param in zip(par.keys(), self.global_model):
                par[key] = torch.from_numpy(param)
            self.net.load_state_dict(par, strict=True)
            sum_accu = 0
            num = 0
            for data, label in self.test_data_loader:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.net(data)
                preds = torch.argmax(preds, dim=1)
                sum_accu += (preds == label).float().mean()
                num += 1
            logger.info(f'round {i} | Global acc: {sum_accu / num}')


if __name__ == '__main__':
    BatchAggrServer().start()
