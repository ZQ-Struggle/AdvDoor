# -*- coding:utf-8 -*-

from conf import *


def print_f1(tp, fn, fp):
    # tp = num_bad_removed
    # fn = num_poisoned_after
    # fp = len(top_scores) - num_bad_removed
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)

    print('precision = {:.2f}'.format(precision *100))
    print('recall = {:.2f}'.format(recall*100))
    print('f1-score = {:.2f}'.format(f1*100))


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(28, 28), grayscale=True)
    input_img_data = image.img_to_array(img)
    input_img_data = input_img_data.reshape(1, 28, 28, 1)

    input_img_data = input_img_data.astype('float32')
    input_img_data /= 255

    # input_img_data = preprocess_input(input_img_data)  # final input shape = (1,224,224,3)
    return input_img_data


def get_signature():
    now = datetime.datetime.now()
    past = datetime.datetime(2015, 6, 6, 0, 0, 0, 0)
    timespan = now - past
    time_sig = int(timespan.total_seconds() * 1000)

    return str(time_sig)


def serialize_img(img, param):
    save_name = '_'.join([param.get_conf('model_prefix'), get_date(), 'image', get_signature()])
    save_path = os.path.join(param.get_conf('perturbation_dir'), save_name + '.png')
    save_pkl = os.path.join(param.get_conf('perturbation_dir'), save_name + '.pkl')

    img = img.flatten().reshape((28, 28))
    print('img.shape = ', img.shape)

    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.show()

    imageio.imwrite(uri=save_path, im=img)

    with open(save_pkl, 'wb') as f:
        pickle.dump(img, f)

    print('save img done')


def deserialize_pert(save_pkl, alpha):
    with open(save_pkl, 'rb') as f:
        perturb = pickle.load(f)

    print('load perturbation done', save_pkl)
    print('self.perturb.shape = {}, magnitude of pert is {}'.format(perturb.shape, np.linalg.norm(perturb)))

    print('alpha = ', alpha)

    perturb = perturb * alpha
    # cilp the float part, 3.7->4, 3.1->3
    # perturb = (perturb*255).astype(np.int32)
    # perturb = perturb.astype(np.uint8)

    return perturb


def to_categorical(labels, nb_classes=None):
    """
    Convert an array of labels to binary class matrix.

    :param labels: An array of integer labels of shape `(nb_samples,)`
    :type labels: `np.ndarray`
    :param nb_classes: The number of classes (possible labels)
    :type nb_classes: `int`
    :return: A binary matrix representation of `y` in the shape `(nb_samples, nb_classes)`
    :rtype: `np.ndarray`
    """
    labels = np.array(labels, dtype=np.int32)
    if not nb_classes:
        nb_classes = np.max(labels) + 1
    categorical = np.zeros((labels.shape[0], nb_classes), dtype=np.float32)
    categorical[np.arange(labels.shape[0]), np.squeeze(labels)] = 1
    return categorical


def preprocess_mnist(x, y, nb_classes=10, clip_values=None):
    """Scales `x` to [0, 1] and converts `y` to class categorical confidences.

    :param x: Data instances.
    :type x: `np.ndarray`
    :param y: Labels.
    :type y: `np.ndarray`
    :param nb_classes: Number of classes in dataset.
    :type nb_classes: `int`
    :param clip_values: Original data range allowed value for features, either one respective scalar or one value per
           feature.
    :type clip_values: `tuple(float, float)` or `tuple(np.ndarray, np.ndarray)`
    :return: Rescaled values of `x`, `y`
    :rtype: `tuple`
    """
    if clip_values is None:
        min_, max_ = np.amin(x), np.amax(x)
    else: 
        min_, max_ = clip_values

    normalized_x = (x - min_) / (max_ - min_)
    categorical_y = to_categorical(y, nb_classes)

    return normalized_x, categorical_y


def preprocess_x_mnist(x, clip_values=None):
    """Scales `x` to [0, 1] and converts `y` to class categorical confidences.

    :param x: Data instances.
    :type x: `np.ndarray`
    :param y: Labels.
    :type y: `np.ndarray`
    :param nb_classes: Number of classes in dataset.
    :type nb_classes: `int`
    :param clip_values: Original data range allowed value for features, either one respective scalar or one value per
                   feature.
    :type clip_values: `tuple(float, float)` or `tuple(np.ndarray, np.ndarray)`
    :return: Rescaled values of `x`, `y`
    :rtype: `tuple`
    """
    if len(x.shape) == 2:
        x = np.expand_dims(x, axis=2)
    if len(x.shape) == 3:
        x = np.expand_dims(x, axis=0)
    if clip_values is None:
        min_, max_ = np.amin(x), np.amax(x)
    else:
        min_, max_ = clip_values

    normalized_x = (x - min_) / (max_ - min_)

    return normalized_x


def preprocess_input_vgg(x):
    if (len(x.shape) == 3):
        x = np.expand_dims(x, axis=0)

    x = x.astype(np.float64)
    x = preprocess_input(x)
    return x


def deprocess_vgg(x):
    x = x.reshape((224, 224, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def load_img(img_path):
    img = cv2.imread(img_path)[:, :, ::-1]
    img = cv2.resize(img, (224, 224))
    img = preprocess_input_vgg(img)
    return img


def dump_model(model, param, prefix='model_prefix'):
    # concat dump name
    serialize_name = '_'.join([param.get_conf()[prefix], get_date()]) + '.pkl'
    print('serialize_name = ', serialize_name)

    # concat dump path
    serialize_path = os.path.join(param.get_conf('save_dir'), serialize_name)
    with open(serialize_path, 'wb') as f:
        pickle.dump(model, f)

    print('model dump success')

    return serialize_path


def deserialize_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)

    print('model load success')

    return model


def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


class Param:
    def __init__(self, json_file):
        self.conf = None
        self.json_file = json_file

    def load_json(self, prefix=None):
        if prefix:
            self.json_path = os.path.join(prefix, self.json_file)
        else:
            self.json_path = os.path.join(json_dir, self.json_file)

        with open(self.json_path, 'r') as f:
            self.conf = json.load(f)

        for key, val in self.conf.items():
            print(key, ':', val)

    def get_conf(self, key_value=None):
        if key_value == None:
            return self.conf
        return self.conf[key_value]

    def set_conf(self, key, value):
        self.conf[key] = value

    def print_conf(self):
        for key, val in self.conf.items():
            print(key, ':', val)
