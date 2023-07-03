# -*- coding: utf-8 -*-
# @Time : 2022/9/26 19:57
# @Author : jklujklu
# @Email : jklujklu@126.com
# @File : dataset.py
# @Software: PyCharm

import os

import numpy as np
from sklearn.model_selection import train_test_split

from utils import load_image, dense_to_one_hot


class GetDataSet(object):
    def __init__(self, train_images_path, test_images_path='', is_iid=False):
        self.train_images_path = train_images_path
        self.test_images_path = test_images_path

        self.IMG_WIDTH = 30
        self.IMG_HEIGHT = 30

    def load_data(self):
        images = list()
        labels = list()
        NUM_CATEGORIES = len(os.listdir(self.train_images_path))
        for category in range(NUM_CATEGORIES):
            categories = os.path.join(self.train_images_path, str(category))
            for img in os.listdir(categories):
                img = load_image(os.path.join(categories, img), target_size=(self.IMG_WIDTH, self.IMG_HEIGHT))
                image = np.array(img)
                images.append(image)
                labels.append(dense_to_one_hot(category, num_classes=43))
        images = np.array(images)
        labels = np.array(labels)
        images = np.transpose(images, (0, 3, 1, 2))
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
        x_train, x_test, y_train, y_test = train_test_split(np.array(images), labels, test_size=0.3)
        return x_train, x_test, y_train, y_test
