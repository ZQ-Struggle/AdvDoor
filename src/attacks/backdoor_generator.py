# -*- coding:utf-8 -*-

import abc

from utils import *


class BackdoorGenerator(metaclass=abc.ABCMeta):
    def __init__(self, model, param):
        self.model = model
        self.param = param

    def serialize(self, postfix='perturbation'):
        self.save_name = '_'.join([self.param.get_conf('model_prefix'), postfix, get_date(),])
        self.save_png = os.path.join(self.param.get_conf('perturbation_dir'), self.save_name + '.png')
        self.save_pkl = os.path.join(self.param.get_conf('perturbation_dir'), self.save_name + '.pkl')
        # self.save_path = os.path.join(self.param.get_conf('perturbation_dir'), self.save_name)

        plt.figure()
        perturb_squeeze = np.squeeze(self.perturb)
        if self.param.get_conf('model_prefix') == 'mnist':
            plt.imshow(perturb_squeeze, cmap='gray')
        else:
            plt.imshow(self.perturb_to_image(self.perturb))
        plt.show()
        print('perturb_squeeze.shape = ', perturb_squeeze.shape)
        print('self.perturb.shape = ', self.perturb.shape)

        imageio.imwrite(uri=self.save_png, im=perturb_squeeze)

        # im_imageio = imageio.imread(uri=self.save_png)
        # print('im_imageio.shape = ', im_imageio.shape)

        with open(self.save_pkl, 'wb') as f:
            pickle.dump(self.perturb, f)

        print('save perturbation done, name = ', self.save_pkl)
        return self.save_pkl

    def predict(self, img):
        pred = self.model.predict_instance(img)
        label = np.argmax(pred[0])

        print('label = ', label)
        print('pred = ', pred)

        return label, pred

    def serialize_img(self, img, postfix='image', is_deprocess=False):
        save_name = '_'.join([self.param.get_conf('model_prefix'), get_date(), postfix, get_signature()]) + '.png'
        save_path = os.path.join(self.param.get_conf('perturbation_dir'), save_name)

        if self.param.get_conf('model_prefix') in models_noLoad:
            # img = np.squeeze(img, axis=(2,))
            img = np.squeeze(img)
            img = np.clip(img * 255, 0, 255)
        elif self.param.get_conf('model_prefix') in models_load:
            # img = img.flatten().reshape((224, 224, 3))
            if is_deprocess:
                img = deprocess_vgg(img)
            img = np.squeeze(img)

        # print('img.shape = ', img.shape)
        print('save_name = ', save_name)

        # perturb_squeeze = np.squeeze(img, axis=(0,))
        imageio.imsave(save_path, img)

        print('save img done')

    def deserialize(self, save_pkl):

        with open(save_pkl, 'rb') as f:
            self.perturb = pickle.load(f)

        # self.perturb = self.perturb.reshape(self.model.get_input_shape)

        print('load perturbation done')
        print('self.perturb.shape = ', self.perturb.shape)

        return self.perturb

    def perturb_to_image(self, x):
        x = x.reshape((self.param.get_conf('train_image_size'),self.param.get_conf('train_image_size'),3))
        # 'BGR'->'RGB'
        x = x[:, :, ::-1]
        # normalize to [0,1] -> [0,255]
        x_normed = ((x - x.min()) / (x.max() - x.min())) * 255
        x_normed = np.clip(x_normed, 0, 255).astype('uint8')
        return x_normed
