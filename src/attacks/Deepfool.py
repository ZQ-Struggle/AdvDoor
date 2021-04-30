# -*- coding:utf-8 -*-


from attacks.backdoor_generator import BackdoorGenerator
from utils import *


class Deepfool(BackdoorGenerator):
    def __init__(self, model, param, pair=None):
        super(Deepfool, self).__init__(model, param)
        if pair is None:
            self.get_loss_gradient(param.get_conf('poison_label_source'), param.get_conf('poison_label_target'))
        else:
            self.get_loss_gradient(pair[0], pair[1])

    def deepfool(self, image, source, target, overshoot=0.02, max_iter=150):
        """
           :param image: Image of size HxWx3
           :param f: feedforward function (input: images, output: values of activation BEFORE softmax).
           :param grads: gradient functions with respect to input (as many gradients as classes).
           :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
           :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
           :param max_iter: maximum number of iterations for deepfool (default = 10)
           :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
        """

        input_shape = image.shape
        pert_image = image

        # iterate = K.function([self.input_tensor], [self.before_softmax_tensor])
        [grads_s, grads_t, f_i, pred] = self.iterate([image])
        f_i = f_i[0]
        pred = pred[0]

        # distance = max(abs(f_i[target] - f_i[source]), 10)
        # distance = max(abs(f_i[target] - f_i[source]), 1)

        # f_i = np.array(f).flatten()
        k_i = int(np.argmax(f_i))

        w = np.zeros(input_shape)
        r_tot = np.zeros(input_shape)

        loop_i = 0
        pert = np.inf
        while (k_i != target or pred[target] < 0.8) and loop_i < max_iter:
            w_k = grads_t - grads_s

            f_k = (f_i[target] - f_i[source]) * 2 #- distance
            pert_k = abs(f_k) / (np.linalg.norm(w_k.flatten(), ord=2)
                                 # * 256.0
                                 )

            # determine which w_k to use

            pert = pert_k
            w = w_k

            # compute r_i and r_tot
            r_i = pert * w / (np.linalg.norm(w.flatten(), ord=2))  # * 256.0)
            r_tot = r_tot + r_i

            # compute new perturbed image
            pert_image = np.clip(image + (1 + overshoot) * r_tot, 0, 1)
            r_tot = (pert_image - image) / (1 + overshoot)
            
            # pert_image = image + (1 + overshoot) * r_tot

            # pert_image = deprocess_vgg(pert_image).astype(np.float64)
            # pert_image = preprocess_input_vgg(pert_image)

            loop_i += 1

            [grads_s, grads_t, f, pred] = self.iterate([pert_image])
            pred = pred[0]

            # compute new label
            f_i = np.array(f).flatten()
            k_i = int(np.argmax(f))

        r_tot = (1 + overshoot) * r_tot

        return r_tot, loop_i, pert_image

    def get_loss_gradient(self, source, target):
        if self.param.get_conf('model_path') == 'origin':
            self.input_tensor = self.model.get_input_tensor_origin()
            self.before_softmax_tensor = self.model.get_before_softmax_tensor_origin()
        else:
            self.input_tensor = self.model.get_classifier().get_input_tensor()
            self.before_softmax_tensor = self.model.get_classifier().get_output_bef_softmax()
            self.output_tensor = self.model.get_classifier().get_output_tensor()

        self.dydx_s = K.gradients(self.before_softmax_tensor[..., source], self.input_tensor)[0]
        self.dydx_t = K.gradients(self.before_softmax_tensor[..., target], self.input_tensor)[0]

        self.iterate = K.function([self.input_tensor],
                                  [self.dydx_s, self.dydx_t, self.before_softmax_tensor, self.output_tensor])

    def gen_perturbation(self, img, source=5, target=6):
        self.perturb, self.loop_i, self.pert_image = self.deepfool(img, source, target)
        self.perturb = np.squeeze(self.perturb)
        return self.perturb, self.loop_i, self.pert_image
