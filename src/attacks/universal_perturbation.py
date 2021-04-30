from attacks.Deepfool import Deepfool
from attacks.backdoor_generator import BackdoorGenerator
from conf import *
from utils import preprocess_input_vgg


class Universal_perturbation(BackdoorGenerator):
    def __init__(self, model, param):
        super(Universal_perturbation, self).__init__(model, param)
        self.deepfool = Deepfool(self.model, self.param)
        self.data_path = os.path.join(self.param.get_conf('data_path'), 'train')
        self.image_size = (self.param.get_conf('train_image_size'), self.param.get_conf('train_image_size'))

    def proj_lp(self, v, xi, p):

        # Project on the lp ball centered at 0 and of radius xi

        # SUPPORTS only p = 2 and p = Inf for now
        if p == 2:
            v = v * min(1, xi / np.linalg.norm(v.flatten(1)))
            # v = v / np.linalg.norm(v.flatten(1)) * xi
        elif p == np.inf:
            v = np.sign(v) * np.minimum(abs(v), xi)
        else:
            raise ValueError('Values of p different from 2 and Inf are currently not supported...')

        return v

    def universal_perturbation(self, dataset, source, target, delta=0.2, max_iter_uni=5, xi=15.0/255.0, p=np.inf,
                               overshoot=0.02, max_iter_df=20):

        """
        :param dataset: Images of size MxHxWxC (M: number of images)

        :param f: feedforward function (input: images, output: values of activation BEFORE softmax).

        :param grads: gradient functions with respect to input (as many gradients as classes).

        :param delta: controls the desired fooling rate (default = 80% fooling rate)

        :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = np.inf)

        :param xi: controls the l_p magnitude of the perturbation (default = 10)

        :param p: norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)

        :param num_classes: num_classes (limits the number of classes to test against, by default = 10)

        :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).

        :param max_iter_df: maximum number of iterations for deepfool (default = 10)

        :return: the universal perturbation.
        """

        v = 0
        fooling_rate = 0.0
        batch_size = self.param.get_conf('batch_size')
        if self.param.get_conf('model_prefix') in models_load:
            source_imgs = dataset.x_train
            source_imgs = [source_imgs[i] for i in np.where(dataset.y_train == source)[0]]
            y_train = dataset.y_train[(dataset.y_train == source)]
        else:
            y_train = dataset.y_train[(dataset.y_train.argmax(axis=1).flatten() == source)]
            source_imgs = dataset.x_train
            source_imgs = [source_imgs[i] for i in np.where(dataset.y_train.argmax(axis=1).flatten() == source)[0]]
            source_imgs = np.array(source_imgs)
        num_images = len(source_imgs)
        num_selection = min(num_images, 5000)

        print('num_selection = ', num_selection)

        itr = 0
        index = np.arange(num_selection)
        while fooling_rate < 1 - delta and itr < max_iter_uni:
            # Shuffle the dataset
            np.random.shuffle(index)

            print('Starting pass number ', itr)

            # Go through the data set and compute the perturbation increments sequentially
            for idx, k in enumerate(index):
                if self.param.get_conf('model_prefix') in models_load:
                    cur_img = source_imgs[k]
                    cur_img = cv2.imread(os.path.join(self.data_path, cur_img))[:, :, ::-1]
                    cur_img = cv2.resize(cur_img, self.image_size)
                    cur_img = preprocess_input_vgg(cur_img)
                else:
                    cur_img = source_imgs[k:k + 1]
                if idx % 1000 == 999:
                        print('>> k = ', idx, ', img_idx = ', k, ', pass #', itr)
                if target != int(np.argmax(np.array(self.deepfool.iterate([cur_img + v])[2]).flatten())):
                    

                    # Compute adversarial perturbation
                    dr, iter, _ = self.deepfool.deepfool(cur_img + v, source=source, target=target, overshoot=overshoot,
                                                         max_iter=max_iter_df)

                    # Make sure it converged...
                    if iter < max_iter_df - 1:
                        v = v + dr

                        # Project on l_p ball
                        v = self.proj_lp(v, xi, p)

            itr = itr + 1
            # v *= 0.99
            est_labels_pert = np.zeros(num_selection)

            num_batches = np.int(np.ceil(np.float(num_selection) / np.float(batch_size)))
            np.random.shuffle(index)
            imgs_test = [source_imgs[i] for i in index[:num_selection]]
            for ii in range(0, num_batches):
                m = (ii * batch_size)
                M = min((ii + 1) * batch_size, num_selection)
                if self.param.get_conf('model_prefix') in models_load:
                    imgs = []
                    for fi in imgs_test[m:M]:
                        img = cv2.imread(os.path.join(self.data_path, fi))[:, :, ::-1]
                        img = cv2.resize(img, self.image_size)
                        imgs.append(img)
                    imgs = preprocess_input_vgg(np.array(imgs))
                else:
                    imgs = np.array(imgs_test[m:M])
                # imgs[:] += v
                imgs = np.clip(imgs[:] + v, 0, 1)
                est_labels_pert[m:M] = np.argmax(self.deepfool.iterate([imgs])[2], axis=1).flatten()
            # Compute the fooling rate
            if self.param.get_conf('model_prefix') in models_load:
                fooling_rate = float(np.sum(est_labels_pert != y_train[index[:num_selection]]) /
                                     float(num_selection))
            else:
                fooling_rate = float(np.sum(est_labels_pert != y_train[index[:num_selection]].argmax(axis=1)) /
                                     float(num_selection))
            print('FOOLING RATE = ', fooling_rate)
        self.perturb = v
        print('magnitude of pert is', np.linalg.norm(v))
        return v
