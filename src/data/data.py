# -*- coding:utf-8 -*-

import abc


class Data(metaclass=abc.ABCMeta):
    def __init__(self, param):
        self.param = param
        self.init()
        self.batch_size = None

    @abc.abstractmethod
    def init(self):
        pass

    @abc.abstractmethod
    def load_data(self, is_add_channel=False):
        pass

    @abc.abstractmethod
    def add_channel_axis(self):
        pass

    @abc.abstractmethod
    def gen_indices(self):
        pass

    @abc.abstractmethod
    def gen_train_data(self):
        pass

    @abc.abstractmethod
    def gen_train_backdoor_data(self):
        pass

    @abc.abstractmethod
    def gen_shuffled_indices(self):
        pass

    @abc.abstractmethod
    def gen_shuffle_train_data(self):
        pass

    @abc.abstractmethod
    def print_backdoor_info(self):
        pass

    @abc.abstractmethod
    def gen_train_backdoor(self):
        pass

    @abc.abstractmethod
    def gen_test_backdoor_data(self):
        pass

    @abc.abstractmethod
    def gen_test_backdoor(self):
        pass

    @abc.abstractmethod
    def gen_backdoor(self, model):
        pass

    @abc.abstractmethod
    def restore_train_backdoor_data(self, poison):
        pass

    @abc.abstractmethod
    def restore_train_backdoor(self, poison):
        pass

    @abc.abstractmethod
    def restore_test_backdoor_data(self, poison):
        pass

    @abc.abstractmethod
    def restore_test_backdoor(self, poison):
        pass

    @abc.abstractmethod
    def restore_backdoor(self, model):
        pass

    @abc.abstractmethod
    def visiualize_img_by_idx(self, shuffled_idx, pre_label, is_train=True):
        pass

    @abc.abstractmethod
    def cal_index(self, idx, is_train=True):
        pass
