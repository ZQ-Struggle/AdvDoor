# MIT License
#
# Copyright (C) IBM Corporation 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Module providing visualization functions.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os.path

from sklearn.manifold import TSNE
from sklearn.datasets import load_iris, load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

from utils import *

# import sys
# sys.path.append(os.path.dirname(__file__) + os.sep + './')

logger = logging.getLogger(__name__)


def create_sprite(images):
    """
    Creates a sprite of provided images.

    :param images: Images to construct the sprite.
    :type images: `np.array`
    :return: An image array containing the sprite.
    :rtype: `np.ndarray`
    """
    shape = np.shape(images)

    if len(shape) < 3 or len(shape) > 4:
        raise ValueError('Images provided for sprite have wrong dimensions ' + str(len(shape)))

    if len(shape) == 3:
        # Check to see if it's mnist type of images and add axis to show image is gray-scale
        images = np.expand_dims(images, axis=3)
        shape = np.shape(images)

    # Change black and white images to RGB
    if shape[3] == 1:
        images = convert_to_rgb(images)

    n = int(np.ceil(np.sqrt(len(images))))
    padding = ((0, n ** 2 - images.shape[0]), (0, 0), (0, 0)) + ((0, 0),) * (images.ndim - 3)
    images = np.pad(images, padding, mode='constant', constant_values=0)

    # Tile the individual thumbnails into an image
    images = images.reshape((n, n) + images.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, images.ndim + 1)))
    images = images.reshape((n * images.shape[1], n * images.shape[3]) + images.shape[4:])

    if images.max() > 2:
        sprite = images
    else:
        sprite = (images * 255).astype(np.uint8)

    return np.array(sprite)


def convert_to_rgb(images):
    """
    Converts grayscale images to RGB. It changes NxHxWx1 to a NxHxWx3 array, where N is the number of figures,
    H is the high and W the width.

    :param images: Grayscale images of shape (NxHxWx1).
    :type images: `np.ndarray`
    :return: Images in RGB format of shape (NxHxWx3).
    :rtype: `np.ndarray`
    """
    dims = np.shape(images)
    if not ((len(dims) == 4 and dims[-1] == 1) or len(dims) == 3):
        raise ValueError('Unexpected shape for grayscale images:' + str(dims))

    if dims[-1] == 1:
        # Squeeze channel axis if it exists
        rgb_images = np.squeeze(images, axis=-1)
    else:
        rgb_images = images
    rgb_images = np.stack((rgb_images,) * 3, axis=-1)

    return rgb_images


def save_image(image, f_name):
    """
    Saves image into a file inside `DATA_PATH` with the name `f_name`.

    :param image: Image to be saved
    :type image: `np.ndarray`
    :param f_name: File name containing extension e.g., my_img.jpg, my_img.png, my_images/my_img.png
    :type f_name: `str`
    :return: `None`
    """
    file_name = os.path.join(clutser_result, f_name)
    folder = os.path.split(file_name)[0]
    if not os.path.exists(folder):
        os.makedirs(folder)

    from PIL import Image
    im = Image.fromarray(image)
    im.save(file_name)
    logger.info('Image saved to %s.', file_name)


def plot_3d(points, labels, colors=None, save=True, f_name=''):
    """
    Generates a 3-D plot in of the provided points where the labels define the
    color that will be used to color each data point.
    Concretely, the color of points[i] is defined by colors(labels[i]).
    Thus, there should be as many labels as colors.

    :param points: arrays with 3-D coordinates of the plots to be plotted
    :type points: `np.ndarray`
    :param labels: array of integers that determines the color used in the plot for the data point.
        Need to start from 0 and be sequential from there on.
    :type labels: `lst`
    :param colors: Optional argument to specify colors to be used in the plot. If provided, this array should contain
    as many colors as labels.
    :type `lst`
    :param save:  When set to True, saves image into a file inside `DATA_PATH` with the name `f_name`.
    :type save: `bool`
    :param f_name: Name used to save the file when save is set to True
    :type f_name: `str`
    :return: fig
    :rtype: `matplotlib.figure.Figure`
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from mpl_toolkits import mplot3d

        if colors is None:
            colors = []
            for i in range(len(np.unique(labels))):
                colors.append('C' + str(i))
        else:
            if len(colors) != len(np.unique(labels)):
                raise ValueError('The amount of provided colors should match the number of labels in the 3pd plot.')

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        for i, coord in enumerate(points):
            try:
                color_point = labels[i]
                ax.scatter3D(coord[0], coord[1], coord[2], color=colors[color_point])
            except IndexError:
                raise ValueError('Labels outside the range. Should start from zero and be sequential there after')
        if save:
            file_name = os.path.realpath(os.path.join(clutser_result, f_name))
            folder = os.path.split(file_name)[0]

            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig(file_name, bbox_inches='tight')
            logger.info('3d-plot saved to %s.', file_name)

        return fig
    except ImportError:
        logger.warning("matplotlib not installed. For this reason, cluster visualization was not displayed.")


def visualize_img_without_backdoor(img, label_org, label_pre, is_train="Train"):
    try:
        import matplotlib
        import matplotlib.pyplot as  plt
    except:
        print("matplotlib not installed. For this reason, cluster visualization was not displayed")
    img = np.squeeze(img)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.axis('off')
    # print(img.shape)
    if len(img.shape) == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.text(0, 0.65, 'data set: ' + is_train, fontsize=20)
    plt.text(0, 0.55, 'original label: ' + str(label_org), fontsize=20)
    plt.text(0, 0.45, 'predicted label: ' + str(label_pre), fontsize=20)
    plt.show()


def save_png(img, idx):
    plt.figure()
    plt.axis('off')
    if len(img.shape) == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)
    plt.savefig('../log/20200115/' + '_'.join([get_date(), get_signature(), str(idx)]) + '.png', format='png')


def save_eps(img_backdoor):
    plt.figure()
    plt.axis('off')
    if len(img_backdoor.shape) == 2:
        plt.imshow(img_backdoor, cmap="gray")
    else:
        plt.imshow(img_backdoor)
    plt.savefig('../log/20191218/' + '_'.join([get_date(), get_signature()]) + '.eps', format='eps')


def visualize_img_with_backdoor(img_orig, label_org, label_pre, img_backdoor, backdoor, is_train='Train'):
    try:
        import matplotlib
        import matplotlib.pyplot as  plt
    except:
        print("matplotlib not installed. For this reason, cluster visualization was not displayed")

    img_orig = np.squeeze(img_orig)
    img_backdoor = np.squeeze(img_backdoor)

    save_eps(img_backdoor)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.axis('off')
    # print(img.shape)
    if len(img_orig.shape) == 2:
        plt.imshow(img_orig, cmap="gray")
    else:
        plt.imshow(img_orig)

    plt.subplot(1, 3, 2)
    plt.axis("off")
    if len(img_backdoor.shape) == 2:
        plt.imshow(img_backdoor, cmap="gray")
    else:
        plt.imshow(img_backdoor)
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.text(0, 0.65, 'data set: ' + is_train, fontsize=18)
    plt.text(0, 0.55, 'original label: ' + str(label_org), fontsize=18)
    plt.text(0, 0.45, 'predicted label: ' + str(label_pre), fontsize=18)
    plt.text(0, 0.35, str(label_org) + " --> " + str(backdoor), fontsize=18)

    plt.show()


def cal_index(self, idx, is_train=True):
    if is_train:
        return idx - len(self.random_selection_indices)
    else:
        return idx - len(self.y_test) + len(self.test_poisoned_index)


def t_sne(digits_data=None, digits_target=None):
    # digits = load_digits()
    # X_tsne = TSNE(n_components=2, random_state=33).fit_transform(digits.data)
    # X_pca = PCA(n_components=2).fit_transform(digits.data)

    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(digits_data)
    X_pca = PCA(n_components=2).fit_transform(digits_data)

    font = {"color": "darkred",
            "size": 13,
            "family": "serif"}

    plt.style.use("ggplot")
    plt.figure(figsize=(8.5, 4))
    plt.subplot(1, 2, 1)

    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=digits.target, alpha=0.6,
    #             cmap=plt.cm.get_cmap('rainbow', 10))

    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=digits_target, alpha=0.6,
                cmap=plt.cm.get_cmap('rainbow', 10))

    plt.title("t-SNE", fontdict=font)
    cbar = plt.colorbar(ticks=range(10))
    cbar.set_label(label='digit value', fontdict=font)
    plt.clim(-0.5, 9.5)
    plt.subplot(1, 2, 2)

    # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=digits.target, alpha=0.6,
    #             cmap=plt.cm.get_cmap('rainbow', 10))

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=digits_target, alpha=0.6,
                cmap=plt.cm.get_cmap('rainbow', 10))

    plt.title("PCA", fontdict=font)
    cbar = plt.colorbar(ticks=range(10))
    cbar.set_label(label='digit value', fontdict=font)
    plt.clim(-0.5, 9.5)
    plt.tight_layout()

    check_dir(tsne_result)

    plt.savefig(os.path.join(tsne_result, '_'.join(['t_sne', get_date(), get_signature()])))
    plt.show()


def t_sne_vis(digits_data=None, digits_target=None):
    # digits = load_digits()
    # X_tsne = TSNE(n_components=2, random_state=33).fit_transform(digits.data)
    # X_pca = PCA(n_components=2).fit_transform(digits.data)


    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(digits_data)

    font = {"color": "darkred",
            "size": 13,
            "family": "serif"}

    plt.style.use("ggplot")
    plt.figure(figsize=(8.5, 8.5))
    # plt.axis('off')
    # plt.subplot(1, 2, 1)

    colors = ['b', 'c', 'y', 'm', 'r']

    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=digits.target, alpha=0.6,
    #             cmap=plt.cm.get_cmap('rainbow', 10))

    lo = plt.scatter(X_tsne[:, 0][np.where(digits_target==0)[0]],
                X_tsne[:, 1][np.where(digits_target==0)[0]],
                alpha=0.6,
                color=colors[0])

    ll = plt.scatter(X_tsne[:, 0][np.where(digits_target == 1)[0]],
                X_tsne[:, 1][np.where(digits_target == 1)[0]],
                alpha=0.6,
                color=colors[1])

    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=digits_target, alpha=0.6,
    #             cmap=plt.cm.get_cmap('rainbow', 10)
    #             )

    plt.legend((lo, ll),
               ('clean','poison'),
               scatterpoints=1,
               loc='upper left')
    # plt.title("t-SNE", fontdict=font)
    # cbar = plt.colorbar(ticks=range(10))
    # cbar.set_label(label='digit value', fontdict=font)
    # plt.clim(-0.5, 9.5)
    # plt.subplot(1, 2, 2)

    # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=digits.target, alpha=0.6,
    #             cmap=plt.cm.get_cmap('rainbow', 10))

    # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=digits_target, alpha=0.6,
    #             cmap=plt.cm.get_cmap('rainbow', 10))

    # plt.title("PCA", fontdict=font)
    # cbar = plt.colorbar(ticks=range(10))
    # cbar.set_label(label='digit value', fontdict=font)
    # plt.clim(-0.5, 9.5)
    # plt.tight_layout()

    check_dir(tsne_result)

    plt.savefig(os.path.join(tsne_result, '_'.join(['t_sne', get_date(), get_signature()])) + '.eps', format='eps')
    plt.show()
    # plt.savefig(os.path.join(tsne_result, '_'.join(['t_sne', get_date(), get_signature()])))


def pca_vis(digits_data=None, digits_target=None):
    X_pca = PCA(n_components=2).fit_transform(digits_data)

    plt.style.use("ggplot")
    plt.figure(figsize=(8.5, 8.5))

    colors = ['b', 'c', 'y', 'm', 'r']

    lo = plt.scatter(X_pca[:, 0][np.where(digits_target == 0)[0]],
                     X_pca[:, 1][np.where(digits_target == 0)[0]],
                     alpha=0.6,
                     color=colors[0])

    ll = plt.scatter(X_pca[:, 0][np.where(digits_target == 1)[0]],
                     X_pca[:, 1][np.where(digits_target == 1)[0]],
                     alpha=0.6,
                     color=colors[1])

    plt.legend((lo, ll),
               ('clean', 'poison'),
               scatterpoints=1,
               loc='upper left')

    check_dir(tsne_result)

    plt.savefig(os.path.join(tsne_result, '_'.join(['pca', get_date(), get_signature()])) + '.eps', format='eps')
    plt.show()
    # plt.savefig(os.path.join(tsne_result, '_'.join(['pca', get_date(), get_signature()])))


def save_visualize_autoencoder(x_test, decoded_imgs):
    n = 10
    for i in range(1, n + 1):
        save_eps(np.squeeze(x_test[i]))
        save_eps(np.squeeze(decoded_imgs[i]))

