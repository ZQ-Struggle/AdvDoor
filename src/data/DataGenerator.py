# -*- coding:utf-8 -*-
import math
import os

import cv2
import keras
import numpy as np

from keras.utils import to_categorical

from utils import preprocess_input_vgg


class DataGenerator(keras.utils.Sequence):

    def __init__(self, x, param, y=None, batch_size=1, shuffle=True, preprocess=preprocess_input_vgg, postfix=None):
        self.batch_size = batch_size
        self.x = x
        self.y = y
        self.indexes = np.arange(len(self.x))
        self.shuffle = shuffle
        self.preprocess = preprocess
        self.batch_size = batch_size
        self.param = param
        if postfix is None:
            self.data_path = self.param.get_conf('data_path')
        else:
            self.data_path = os.path.join(self.param.get_conf('data_path'), postfix)
        self.train_size = param.get_conf('train_image_size')
        self.nb_class = self.param.get_conf('classes_num')

    def __len__(self):
        return math.ceil(len(self.x) / float(self.batch_size))

    def __getitem__(self, index):
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_datas = [self.x[k] for k in batch_indexs]
        y=None
        if self.y is not None:
            y = [self.y[k] for k in batch_indexs]
        return self.data_generation(batch_datas,y)

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_datas, y=None):
        images = []
        labels = []
        for i, data in enumerate(batch_datas):
            
            img = cv2.imread(os.path.join(self.data_path, data))[:, :, ::-1]
            img = cv2.resize(img, (self.train_size, self.train_size))

            images.append(img)
        if self.y is not None:
            for label in y:
                labels.append(to_categorical(label, self.nb_class))

        images = np.array(images)

        if self.preprocess is not None:
            images = self.preprocess(images)
        # print(images.shape)
        if self.y is None:
            return np.array(images)

        return np.array(images), np.array(labels)

