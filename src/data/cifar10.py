# -*- coding:utf-8 -*-

from keras.datasets import cifar10

from backdoor import Backdoor
from data.data import Data
from utils import *
from visualization import visualize_img_without_backdoor, visualize_img_with_backdoor


class CifarData(Data):

    def __init__(self, param):
        super(CifarData, self).__init__(param)

    def init(self):
        self.x_train_ordered = None
        self.y_train_ordered = None
        self.x_test = None
        self.y_test = None
        self.min_ = None
        self.max_ = None
        self.random_selection_indices = None
        self.train_poisoned_index = None
        self.test_poisoned_index = None
        self.data_path = self.param.get_conf('data_path')
        self.backdoor = Backdoor(self.param.get_conf())
        if type(self.param.get_conf('poison_label_source')) is list:
            self.source_num = self.param.get_conf('poison_label_source')
        else:
            self.source_num = np.array([int(self.param.get_conf('poison_label_source'))])
        if type(self.param.get_conf('poison_label_target')) is list:
            self.target_num = self.param.get_conf('poison_label_target')

        else:
            self.target_num = np.array([int(self.param.get_conf('poison_label_target'))])
 
    def load_data(self, is_add_channel=False):
        # print('self.data_path = ', self.data_path)

        (self.x_train_ordered, self.y_train_ordered), (self.x_test, self.y_test) = cifar10.load_data()
        self.y_train_ordered = self.y_train_ordered.squeeze()
        self.y_test = self.y_test.squeeze()
        # serialize_img(self.x_test[0], self.param)

        # Add channel axis
        self.min_, self.max_ = 0, 255

        self.n_train = np.shape(self.x_train_ordered)[0]

        if is_add_channel:
            self.gen_indices()
            self.shuffled_indices = np.arange(min(len(self.x_train_ordered), self.param.get_conf('num_selection')))
            self.gen_train_data()  # train_data got
            self.add_channel_axis()

        print('after reading data')
        print('x_train.shape = ', self.x_train_ordered.shape)
        print('y_train.shape = ', self.y_train_ordered.shape)
        print('x_test.shape = ', self.x_test.shape)
        print('y_test.shape = ', self.y_test.shape)

    def add_channel_axis(self):
        self.is_poison_train_ordered = np.zeros_like(self.y_train_ordered) == 1
        self.is_poison_test = np.zeros_like(self.y_test) == 1 
        self.x_train_ordered, self.y_train_ordered = preprocess_mnist(self.x_train_ordered,
                                                                      self.y_train_ordered)
        self.x_test, self.y_test = preprocess_mnist(self.x_test, self.y_test)

    def gen_indices(self):
        # self.param.get_conf('num_selection') means number of input case selected
        # self.n_train means total number of input case
        # self.random_selection_indices = np.random.choice(self.n_train, self.param.get_conf('num_selection'))
        self.random_selection_indices = np.arange(self.n_train)
        np.random.shuffle(self.random_selection_indices)
        self.random_selection_indices = self.random_selection_indices[:self.param.get_conf('num_selection')]

    def gen_train_data(self):
        # data.n_train = 60000
        # param.get_conf('num_selection') = 5000
        # random_selection_indices = np.random.choice(self.n_train, self.param.get_conf('num_selection'))

        # update random train data
        self.x_train_ordered = self.x_train_ordered[self.random_selection_indices]
        self.y_train_ordered = self.y_train_ordered[self.random_selection_indices]

    def gen_train_backdoor_data(self):
        # start creating backdoor data
        # the backdoor method can be changed

        self.is_poison_train_ordered, \
        self.x_poisoned_raw, \
        self.y_poisoned_raw = self.backdoor.generate_backdoor(self.x_train_ordered,
                                                              self.y_train_ordered,
                                                              self.backdoor.train_poison_rate,
                                                              sources=self.source_num,
                                                              targets=self.target_num)

        self.x_train_ordered, self.y_train_ordered = preprocess_mnist(self.x_poisoned_raw, self.y_poisoned_raw)

        # Add channel axis:
        # self.x_train_ordered = np.expand_dims(self.x_train_ordered, axis=3)

    def gen_shuffled_indices(self):
        # Shuffle training data so poison is not together
        n_train = np.shape(self.y_train_ordered)[0]
        self.shuffled_indices = np.arange(n_train)
        np.random.shuffle(self.shuffled_indices)

    def gen_shuffle_train_data(self):

        # self.x_train_ordered = self.x_train_ordered[self.shuffled_indices]
        # self.y_train_ordered = self.y_train_ordered[self.shuffled_indices]
        # self.is_poison_train_ordered = self.is_poison_train_ordered[self.shuffled_indices]

        self.is_clean_ordered = (self.is_poison_train_ordered == 0)

    def print_backdoor_info(self,info):
        print('after',info,'backdoor')
        print('x_train.shape = ', self.x_train_ordered.shape)
        print('y_train.shape = ', self.y_train_ordered.shape)
        print('x_poisoned_raw.shape = ', self.x_poisoned_raw.shape)
        print('y_poisoned_raw.shape = ', self.y_poisoned_raw.shape)

        '''
        after generating backdoor
        x_train.shape =  (5000, 28, 28)
        y_train.shape =  (5000,)
        x_poisoned_raw.shape =  (5209, 28, 28)
        y_poisoned_raw.shape =  (5209,)

        5000 -> 5209

        increasing number depends on the number of cases of sources
        generate extra test case from sources to targets
        '''

    def gen_train_backdoor(self):
        self.gen_indices()
        self.gen_train_data()  # train_data got
        self.gen_train_backdoor_data()

        self.gen_shuffled_indices()
        self.gen_shuffle_train_data()
        # print("self.shuffled_indices = ", self.shuffled_indices)

    # test data
    def gen_test_backdoor_data(self):
        # Poison test data
        self.is_poison_test, \
        self.x_poisoned_raw_test, \
        self.y_poisoned_raw_test = self.backdoor.generate_backdoor(self.x_test,
                                                                   self.y_test,
                                                                   self.backdoor.test_poison_rate,
                                                                   sources=self.source_num,
                                                                   targets=self.target_num)

        self.x_test, self.y_test = preprocess_mnist(self.x_poisoned_raw_test, self.y_poisoned_raw_test)
        # Add channel axis:
        # self.x_test = np.expand_dims(self.x_test, axis=3)

    def gen_test_backdoor(self):
        self.gen_test_backdoor_data()

        self.print_backdoor_info("generate")

    def gen_backdoor(self, model=None):
        # self.gen_indices()

        self.gen_train_backdoor()
        # 1. input case index
        # 2. train poison meta data
        # should be stored in model
        self.backdoor.get_poison().set_random_selection_indices(self.random_selection_indices)
        self.backdoor.get_poison().set_shuffled_indices(self.shuffled_indices)
        self.train_poisoned_index = self.backdoor.get_poison().get_indices_to_be_poisoned()
        if model:
            model.set_train_poison(self.backdoor.get_poison())

        self.gen_test_backdoor()

        # test poison meta data
        self.test_poisoned_index = self.backdoor.get_poison().get_indices_to_be_poisoned()
        if model:
            model.set_test_poison(self.backdoor.get_poison())

    # restore train data
    def restore_train_backdoor_data(self, poison):
        self.is_poison_train_ordered, \
        self.x_poisoned_raw, \
        self.y_poisoned_raw = self.backdoor.restore_backdoor(self.x_train_ordered,
                                                             self.y_train_ordered,
                                                             poison)
        # Add channel axis:
        # self.x_poisoned_raw = np.expand_dims(self.x_poisoned_raw, axis=3)
        self.x_train_ordered, self.y_train_ordered = preprocess_mnist(self.x_poisoned_raw, self.y_poisoned_raw)

    def restore_train_backdoor(self, poison):
        self.random_selection_indices = poison.get_random_selection_indices()
        self.shuffled_indices = poison.get_shuffled_indices()
        self.train_poisoned_index = poison.get_indices_to_be_poisoned()
        self.gen_train_data()
        self.restore_train_backdoor_data(poison)

        self.gen_shuffle_train_data()

    def restore_test_backdoor_data(self, poison):
        # Poison test data
        self.is_poison_test, \
        self.x_poisoned_raw_test, \
        self.y_poisoned_raw_test = self.backdoor.restore_backdoor(self.x_test,
                                                                  self.y_test,
                                                                  poison)

        self.x_test, self.y_test = preprocess_mnist(self.x_poisoned_raw_test, self.y_poisoned_raw_test)

        # Add channel axis:
        # self.x_test = np.expand_dims(self.x_test, axis=3)

    @property
    def x_train(self):
        return self.x_train_ordered[self.shuffled_indices]

    @property
    def y_train(self):
        return self.y_train_ordered[self.shuffled_indices]

    @property
    def is_poison_train(self):
        return self.is_poison_train_ordered[self.shuffled_indices]

    @property
    def is_clean(self):
        return self.is_clean_ordered[self.shuffled_indices]

    def restore_test_backdoor(self, poison):
        self.test_poisoned_index = poison.get_indices_to_be_poisoned()
        self.restore_test_backdoor_data(poison)
        self.print_backdoor_info("restore")


    def restore_backdoor(self, model):
        self.restore_train_backdoor(model.get_train_poison())
        self.restore_test_backdoor(model.get_test_poison())

    def get_backdoor(self):
        return self.backdoor

    def set_backdoor(self, backdoor):
        self.backdoor = backdoor

    def visiualize_img_by_idx(self, shuffled_idx, pre_label, is_train=True):

        if is_train:
            idx = self.shuffled_indices[shuffled_idx]
            if self.is_poison_train_ordered[idx]:
                # print("idx of poison in train set", self.train_poisoned_index[self.cal_index(idx)], self.cal_index(idx))
                visualize_img_with_backdoor(
                    self.x_poisoned_raw[self.train_poisoned_index[self.cal_index(idx)]],
                    self.y_train_ordered[self.train_poisoned_index[self.cal_index(idx)]].argmax(),
                    pre_label,
                    self.x_poisoned_raw[idx],
                    np.argmax(self.y_train_ordered[idx])
                )
            else:
                visualize_img_without_backdoor(
                    self.x_poisoned_raw[idx],
                    self.y_train_ordered[idx].argmax(),
                    pre_label,
                    None)
        else:
            idx = shuffled_idx
            if self.is_poison_test[idx]:
                # print("idx of poison in test set", self.test_poisoned_index[self.cal_index(idx, False)],
                #       self.cal_index(idx, False))
                visualize_img_with_backdoor(
                    self.x_poisoned_raw_test[self.test_poisoned_index[self.cal_index(idx, False)]],
                    self.y_test[self.test_poisoned_index[self.cal_index(idx, False)]].argmax(),
                    pre_label,
                    self.x_poisoned_raw_test[idx],
                    np.argmax(self.y_test[idx]),
                    "Test")
                
            else:
                visualize_img_without_backdoor(self.x_poisoned_raw_test[idx], self.y_test[idx].argmax(), pre_label,
                                                "Test" )

    def cal_index(self, idx, is_train=True):
        if is_train:
            return idx - len(self.random_selection_indices)
        else:
            return idx - len(self.y_test) + len(self.test_poisoned_index)

    def get_clean_data(self):
        if not hasattr(self, "is_poison_train"):
            return self.x_train, self.y_train, self.x_test, self.y_test
        return self.x_train[self.is_poison_train == 0], \
               self.y_train[self.is_poison_train == 0], \
               self.x_test[self.is_poison_test == 0], \
               self.y_test[self.is_poison_test == 0]

    def get_poison_data(self):
        return self.x_train[self.is_poison_train == 1], \
               self.y_train[self.is_poison_train == 1], \
               self.x_test[self.is_poison_test == 1], \
               self.y_test[self.is_poison_test == 1]

    def get_specific_label_clean_data(self, label):
        x_train, y_train, x_test, y_test = self.get_clean_data()
        y_train_label = np.argmax(y_train, axis=1)
        y_test_label = np.argmax(y_test, axis=1)
        return x_train[y_train_label == label], \
               y_train[y_train_label == label], \
               x_test[y_test_label == label], \
               y_test[y_test_label == label]

    def get_specific_label_poison_data(self, label):
        x_train, y_train, x_test, y_test = self.get_poison_data()
        y_train_label = np.argmax(y_train, axis=1)
        y_test_label = np.argmax(y_test, axis=1)
        return x_train[y_train_label == label], \
               y_train[y_train_label == label], \
               x_test[y_test_label == label], \
               y_test[y_test_label == label]

    def get_specific_label_data(self, label):
        y_train = self.y_train.argmax(axis=1)
        y_test = self.y_test.argmax(axis=1)
        return self.x_train[y_train == label], \
               self.y_train[y_train == label], \
               self.x_test[y_test == label], \
               self.y_test[y_test == label], \
               self.is_poison_train[y_train == label], \
               self.is_poison_test[y_test == label]


if __name__ == '__main__':
    json_name = sys.argv[1]
    param = Param(json_name)
    param.load_json()
    data = Data(param)
    data.load_data()
    data.gen_backdoor()
