# -*- coding:utf-8 -*-

import numpy as np

from utils import *
import copy


class Backdoor:
    def __init__(self, conf):
        self.train_poison_rate = conf['train_poison_rate']
        self.test_poison_rate = conf['test_poison_rate']
        self.backdoor_type = conf['backdoor_type']
        self.pert_path = conf['pert_path']
        self.poison = None
        self.pert = None
        self.conf = conf
        self.distortion = []
    def generate_backdoor(self, 
                          x_clean,
                          y_clean,
                          percent_poison,
                          sources=np.arange(10),
                          targets=(np.arange(10) + 1) % 10,
                          data_dir=None):
        """
        Creates a backdoor in MNIST images by adding a pattern or pixel to the image and changing the label to a targeted
        class. Default parameters poison each digit so that it gets classified to the next digit.

        :param x_clean: Original raw data
        :type x_clean: `np.ndarray`
        :param y_clean: Original labels
        :type y_clean:`np.ndarray`
        :param percent_poison: After poisoning, the target class should contain this percentage of poison
        :type percent_poison: `float`
        :param backdoor_type: Backdoor type can be `pixel` or `pattern`.
        :type backdoor_type: `str`
        :param sources: Array that holds the source classes for each backdoor. Poison is
        generating by taking images from the source class, adding the backdoor trigger, and labeling as the target class.
        Poisonous images from sources[i] will be labeled as targets[i].
        :type sources: `np.ndarray`
        :param targets: This array holds the target classes for each backdoor. Poisonous images from sources[i] will be
                        labeled as targets[i].
        :type targets: `np.ndarray`
        :return: Returns is_poison, which is a boolean array indicating which points are poisonous, poison_x, which
        contains all of the data both legitimate and poisoned, and poison_y, which contains all of the labels
        both legitimate and poisoned.
        :rtype: `tuple`
        """

        y_poison = np.copy(y_clean)
        is_poison = np.zeros(np.shape(y_poison))

        # for i, (src, tgt) in enumerate(zip(sources, targets)):
        
        

        # num_poison = round((percent_poison * n_points_in_tgt) / (1 - percent_poison))
        if type(sources) is list:
            y_clean_position = []
            n_points_in_src = 0
            for sou in sources:
                n_points_in_src += np.sum(y_clean == sou)
                y_clean_position.append(np.where(y_clean == sou)[0])
            y_clean_position = np.concatenate(y_clean_position, axis=0)
        else:
            n_points_in_src = np.sum(y_clean == sources)
            y_clean_position = np.where(y_clean == sources)[0]
        if type(targets) is list:
            n_points_in_tgt = 0
            for tar in targets:
                n_points_in_tgt += np.sum(y_clean == tar)
        else:
            n_points_in_tgt = np.sum(y_clean == targets)
                
        self.percent_poison = percent_poison
        self.n_points_in_tgt = n_points_in_tgt
        self.n_points_in_src = n_points_in_src
        # self.backdoor_type = backdoor_type

        self.sources = sources
        self.targets = targets

        # generate
        # 1. number of poison
        # 2. indices to be poisoned

        
        self.gen_posion(y_clean_position)

        if isinstance(x_clean[0], str):
            x_poison = x_clean
            imgs_p = [x_clean[i] for i in self.poison.get_indices_to_be_poisoned()]
            inds_save = np.setdiff1d(np.arange(len(x_clean)), self.poison.get_indices_to_be_poisoned())
            x_poison = x_poison[inds_save]
            for f in imgs_p:
                # BGR->RGB
                img = cv2.imread(os.path.join(data_dir, f))[:, :, ::-1]
                img = cv2.resize(img, (self.conf['train_image_size'], self.conf['train_image_size']))
                img = preprocess_input_vgg(img)
                img = self.add_backdoor_on_imgs(img)
                img = deprocess_vgg(img)
                poison_f = f[:-4] + '_poison' + f[-4:]
                poison_f = os.path.join(self.conf['poison_target_name'], os.path.split(poison_f)[1])
                x_poison.append(poison_f)
                # RGB->BGR
                cv2.imwrite(os.path.join(data_dir, poison_f), img[:, :, ::-1])
                
        else:
            x_poison = np.copy(x_clean)
            imgs_p = np.copy(x_clean[self.poison.get_indices_to_be_poisoned()])
            max_val = np.max(x_clean)
            # inds_save = np.setdiff1d(np.arange(len(x_clean)), self.poison.get_indices_to_be_poisoned())
            imgs_to_be_poisoned = self.add_backdoor_on_imgs(imgs_p)
            # x_poison = x_poison[inds_save]
            x_poison = np.append(x_poison, imgs_to_be_poisoned, axis=0)
        # label_p = np.copy(y_clean[self.poison.get_indices_to_be_poisoned()])
        y_poison = np.append(y_poison, np.ones((self.poison.get_num_poison())) * self.targets)
        is_poison = np.append(is_poison, np.ones(self.poison.get_num_poison()))

        is_poison = is_poison != 0

        return is_poison, x_poison, y_poison

    # restore poison from serialized model
    def restore_backdoor(self,
                         x_clean,
                         y_clean,
                         poison,
                         data_dir=None):
        if isinstance(x_clean[0], str):
            imgs_poison = [x_clean[i] for i in poison.get_indices_to_be_poisoned()]
            x_poison = x_clean
            imgs_to_be_poisoned = []
            is_poison = np.zeros(np.shape(y_clean), dtype=np.int32)
            for f in imgs_poison:
                poison_f = f[:-4] + '_poison' + f[-4:]
                poison_f = os.path.join(self.conf['poison_target_name'], os.path.split(poison_f)[1])
                if not os.path.exists(os.path.join(data_dir, poison_f)):
                    img = cv2.imread(os.path.join(data_dir, f))[:, :, ::-1]
                    img = cv2.resize(img, (self.conf['train_image_size'], self.conf['train_image_size']))
                    img = preprocess_input_vgg(img)
                    img = self.add_backdoor_on_imgs(img)
                    img = deprocess_vgg(img)
                    cv2.imwrite(os.path.join(data_dir, poison_f), img[:, :, ::-1])
                imgs_to_be_poisoned.append(poison_f)

            inds_save = np.setdiff1d(np.arange(len(x_clean)), poison.get_indices_to_be_poisoned())
            x_poison = x_poison[inds_save]
            x_poison += imgs_to_be_poisoned
            y_poison = np.append(y_clean, np.ones(poison.get_num_poison()) * poison.get_targets(), axis=0)
            is_poison = np.append(is_poison, np.ones(poison.get_num_poison()))
        else:
            x_poison = np.copy(x_clean)
            if len(y_clean.shape) == 1:
                y_poison = np.copy(y_clean)
            else:
                y_poison = np.argmax(y_clean, axis=1)

            is_poison = np.zeros(np.shape(y_poison))
            # print(y_clean)
            # print(poison.get_sources())

            # no need for generate poison
            # we get poison from serialized model directly

            imgs_to_be_poisoned = np.copy(x_clean[poison.get_indices_to_be_poisoned()])
            # inds_save = np.setdiff1d(np.arange(len(x_clean)), poison.get_indices_to_be_poisoned())
            imgs_to_be_poisoned = self.add_backdoor_on_imgs(imgs_to_be_poisoned)
            # label_p = np.copy(y_clean[self.poison.get_indices_to_be_poisoned()])
            # x_poison = x_poison[inds_save]
            x_poison = np.append(x_poison, imgs_to_be_poisoned, axis=0)
            y_poison = np.append(y_poison, np.ones(poison.get_num_poison()) * poison.get_targets(), axis=0)
            is_poison = np.append(is_poison, np.ones(poison.get_num_poison()))

        is_poison = is_poison != 0

        return is_poison, x_poison, y_poison

    def add_backdoor_on_imgs(self, imgs_to_be_poisoned, max_val=255):
        if self.backdoor_type == 'pattern':
            imgs_to_be_poisoned = self.add_pattern_bd(x=imgs_to_be_poisoned, pixel_value=max_val)
        elif self.backdoor_type == 'pixel':
            imgs_to_be_poisoned = self.add_single_bd(x=imgs_to_be_poisoned, pixel_value=max_val)
        elif self.backdoor_type == 'adversarial':
            # load perturbation
            if self.pert is None:
                self.pert = deserialize_pert(self.pert_path, self.conf['alpha_pert'])

                if self.conf['model_prefix'] in models_noLoad:
                    self.pert = (self.pert * 255).astype(np.int32)
                

            imgs_to_be_poisoned = self.add_adversarial_perturbation(x=imgs_to_be_poisoned)
        return imgs_to_be_poisoned

    def gen_posion(self, y_idx):
        num_poison = int(self.percent_poison * self.n_points_in_tgt)
        num_poison = min(num_poison, self.n_points_in_src)
        indices_to_be_poisoned = np.arange(self.n_points_in_src)
        np.random.shuffle(indices_to_be_poisoned)
        indices_to_be_poisoned = y_idx[indices_to_be_poisoned[:num_poison]]
        self.poison = Poison(num_poison,
                             indices_to_be_poisoned,
                             self.backdoor_type,
                             self.sources,
                             self.targets,
                             self.percent_poison)

    def add_single_bd(self, x, distance=2, pixel_value=1):
        """
        Augments a matrix by setting value some `distance` away from the bottom-right edge to 1. Works for single images
        or a batch of images.
        :param x: N X W X H matrix or W X H matrix. will apply to last 2
        :type x: `np.ndarray`

        :param distance: distance from bottom-right walls. defaults to 2
        :type distance: `int`

        :param pixel_value: Value used to replace the entries of the image matrix
        :type pixel_value: `int`

        :return: augmented matrix
        :rtype: `np.ndarray`
        """
        x = np.array(x)
        shape = x.shape
        if len(shape) == 4:
            width, height = x.shape[1:3]
            # x[:, width - distance, height - distance] = pixel_value
            x= x.astype(np.int32)
            x[:,0::2,0::2,:] += 5
            x = np.clip(x,0,255)
            x=x.astype(np.uint8)
        elif len(shape) == 3:
            width, height,c = x.shape
            x[width - distance, height - distance,:] = pixel_value
        else:
            raise RuntimeError('Do not support numpy arrays of shape ' + str(shape))
        return x

    def add_pattern_bd(self, x, distance=4, pixel_value=1):
        """
        Augments a matrix by setting a checkboard-like pattern of values some `distance` away from the bottom-right
        edge to 1. Works for single images or a batch of images.
        :param x: N X W X H matrix or W X H matrix. will apply to last 2
        :type x: `np.ndarray`
        :param distance: distance from bottom-right walls. defaults to 2
        :type distance: `int`
        :param pixel_value: Value used to replace the entries of the image matrix
        :type pixel_value: `int`
        :return: augmented matrix
        :rtype: np.ndarray
        """
        x = np.array(x)
        shape = x.shape
        if len(shape) == 4:
            width, height = x.shape[1:-1]
            # x[:, width - distance, height - distance,:] = pixel_value
            # x[:, width - distance - 1, height - distance - 1,:] = pixel_value
            # x[:, width - distance , height - distance - 1,:] = pixel_value
            # x[:, width - distance - 1, height - distance,:] = pixel_value
            # x[:, width - distance, height - distance - 2,:] = pixel_value
            # x[:, width - distance - 2, height - distance,:] = pixel_value
            if self.conf['model_prefix'] == "GTSRB":
                x[:, width-distance - 2 : width-distance + 2, height - distance - 2 : height - distance + 2, :] =  [255,255,0]
                # x[:, width-distance - 2 : width-distance + 2, height - distance - 2 : height - distance + 2, 2] = pixel_value
                distance = 15
                # x[:, width-distance + 9: width-distance + 12, height - distance - 2 : height - distance + 1, 0:2] = pixel_value
                # x[:, width-distance + 9 : width-distance + 12, height - distance - 2 : height - distance + 1, :] = [255, 255, 0]
            else:
                x[:, width-distance - 2 : width-distance + 2, height - distance - 2 : height - distance + 2,:] = pixel_value
        elif len(shape) == 3:
            width, height = x.shape[1:]
            x[:, width - distance, height - distance] = pixel_value
            x[:, width - distance - 1, height - distance - 1] = pixel_value
            x[:, width - distance, height - distance - 2] = pixel_value
            x[:, width - distance - 2, height - distance] = pixel_value
        elif len(shape) == 2:
            width, height = x.shape
            x[width - distance, height - distance] = pixel_value
            x[width - distance - 1, height - distance - 1] = pixel_value
            x[width - distance, height - distance - 2] = pixel_value
            x[width - distance - 2, height - distance] = pixel_value
        else:
            raise RuntimeError('Do not support numpy arrays of shape ' + str(shape))
        return x

    def add_adversarial_perturbation(self, x):
        
        x = np.array(x)
        origin_x = copy.copy(x)
        '''
        for i in range(20):
            save_png(np.squeeze(x[i]), i)
        '''
        # x = x.astype(np.int32)
        shape = x.shape

        if self.conf['model_prefix'] in models_noLoad:
            x = x.astype(np.int32)

        if len(shape) == 3:
            # x.shape = (140,28,28)
            # self.pert.shape = (1,28,28,1)
            x[:, ] += np.squeeze(self.pert)
        elif len(shape) == 2:
            # x.shape = (140,28,28)
            # self.pert.shape = (1,28,28,1)
            x += self.pert
        elif len(shape) == 4:
            x += self.pert
        # make sure the value range [0,255]

        if self.conf['model_prefix'] in models_noLoad:
            x = np.clip(x, 0, 255)
        # dis = np.abs(x - origin_x)
        # self.distortion.append(dis)
        '''
        for i in range(20):
            save_png(np.squeeze(x[i]), i)
        '''

        return x #, dis

    def get_poison(self):
        return self.poison

    def set_poison(self, poison):
        self.poison = poison

    def get_pert_path(self):
        return self.pert_path

    def set_pert_path(self, pert_path):
        self.pert_path = pert_path
