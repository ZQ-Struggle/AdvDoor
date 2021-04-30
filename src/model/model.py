# -*- coding:utf-8 -*-

from conf import *
from utils import *
import abc



class CNNModel(metaclass=abc.ABCMeta):
    def __init__(self, param):
        # input_shape = x_train.shape[1:]
        self.param = param
        self.train_poison = None
        self.test_poison = None
        self.classifier = None
    def init(self, data):
        self.input_shape = data.x_train.shape[1:]
        self.min_ = data.min_
        self.max_ = data.max_

    def set_learning_phase(self, learning_phase):
        K.set_learning_phase(learning_phase)

    @abc.abstractmethod
    def init_model(self):
        pass

    def predict_acc(self, x, y, is_poison, type_str):
        # Evaluate the classifier on the test set
        self.test_preds = np.argmax(self.classifier.predict(x), axis=1)
        self.test_acc = np.sum(self.test_preds == np.argmax(y, axis=1)) / y.shape[0]
        print("\n%s accuracy: %.2f%%" % (type_str, self.test_acc * 100))

        # Evaluate the classifier on poisonous data in test set
        # self.poison_preds = np.argmax(self.classifier.predict(x[is_poison]), axis=1)
        self.poison_preds = self.test_preds[is_poison]
        self.poison_acc = np.sum(self.poison_preds == np.argmax(y[is_poison], axis=1)) / max(is_poison.sum(),1)
        print("\nPoisonous %s set accuracy (i.e. effectiveness of poison): %.2f%%" % (type_str, self.poison_acc * 100))

        # Evaluate the classifier on clean data
        # self.clean_preds = np.argmax(self.classifier.predict(x[is_poison == 0]), axis=1)
        self.clean_preds = self.test_preds[is_poison==0]
        self.clean_acc = np.sum(self.clean_preds == np.argmax(y[is_poison == 0], axis=1)) / y[is_poison == 0].shape[0]
        print("\nClean %s set accuracy: %.2f%%" % (type_str, self.clean_acc * 100))

        # when result_dict is not empty, start record experiment results

    # to validate backdoor insert effectiveness
    # check whether the backdoor data with poison label is predicted by the model with poison label
    def predict(self, data):
        # Evaluate the classifier on the train set
        self.predict_acc(data.x_train, data.y_train, data.is_poison_train, 'train')


        # visualize predict
        # for i in range(3):
        #     data.visiualize_img_by_idx(np.where(np.array(data.is_poison_train) == 1)[0][i], self.poison_preds[i])


        # Evaluate the classifier on the test set
        self.predict_acc(data.x_test, data.y_test, data.is_poison_test, 'test')

        '''
        # visualize predict
        for i in range(3):
            print(np.where(np.array(data.is_poison_test) == 1)[0][i])
            data.visiualize_img_by_idx(np.where(np.array(data.is_poison_test) == 1)[0][i], self.poison_preds[i], False)
        '''

    def predict_robust(self, x, y,  is_poison, type_str=''):
        self.test_preds = np.argmax(self.classifier.predict(x), axis=1)
        self.test_acc = np.sum(self.test_preds == np.argmax(y, axis=1)) / y.shape[0]
        print("\n%s accuracy: %.2f%%" % (type_str, self.test_acc * 100))

        # Evaluate the classifier on poisonous data in test set
        # self.poison_preds = np.argmax(self.classifier.predict(x[is_poison]), axis=1)
        self.poison_preds = self.test_preds[is_poison]
        self.poison_acc = np.sum(self.poison_preds == np.argmax(y[is_poison], axis=1)) / max(is_poison.sum(),1)
        print("\nPoisonous %s set accuracy (i.e. effectiveness of poison): %.2f%%" % (type_str, self.poison_acc * 100))

        # Evaluate the classifier on clean data
        # self.clean_preds = np.argmax(self.classifier.predict(x[is_poison == 0]), axis=1)
        self.clean_preds = self.test_preds[is_poison==0]
        self.clean_acc = np.sum(self.clean_preds == np.argmax(y[is_poison == 0], axis=1)) / y[is_poison == 0].shape[0]
        print("\nClean %s set accuracy: %.2f%%" % (type_str, self.clean_acc * 100))
    
    def set_param(self, param):
        self.classifier.param = param
        self.param = param
    
    def get_train_poison(self):
        return self.train_poison

    def set_train_poison(self, poison):
        self.train_poison = poison

    def get_test_poison(self):
        return self.test_poison

    def set_test_poison(self, poison):
        self.test_poison = poison


    def predict_instance(self, x):
        return self.classifier.predict(x)[0]

    def get_input_shape(self):
        return self.input_shape

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape

    def get_classifier(self):
        return self.classifier

    def set_classifier(self, classifier):
        self.classifier = classifier

    def get_input_tensor(self):
        return self.classifier.get_input_tensor()

    def get_output_tensor(self):
        return self.classifier.get_output_tensor()

    @abc.abstractmethod
    def get_dense_tensor(self):
        pass

    