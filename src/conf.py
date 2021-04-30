# -*- coding:utf-8 -*-

import os
import sys
import json
import pickle
import pprint
import imageio
import datetime
import numpy as np
# import pandas as pd
import logging
import logging.config
from poison import *
import cv2
import matplotlib.pyplot as plt
# import seaborn as sns
plt.switch_backend('agg')
import keras.backend as K
from tqdm import *
from numpy import float32
from keras.models import Sequential, Model, load_model
from keras.preprocessing import image
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Activation, Input, UpSampling2D, AveragePooling2D,BatchNormalization
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.callbacks import Callback
models_load = ['vgg16', 'nuaa']
models_noLoad = ['cifar', 'mnist', "GTSRB"]
data_dir = '../data'
json_dir = '../json'
clutser_result = '../vis/clustering_result'
tsne_result = '../vis/t_sne'
MODEL_RESTORE_PATH = '../model/mnist_universal'

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'std': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M'
        }
    },
    'handlers': {
        'default': {
            'class': 'logging.NullHandler',
        },
        'test': {
            'class': 'logging.StreamHandler',
            'formatter': 'std',
            'level': logging.DEBUG
        }
    },
    'loggers': {
        '': {
            'handlers': ['default']
        },
        'testLogger': {
            'handlers': ['test'],
            'level': 'INFO',
            'propagate': True
        }
    }
}
logging.config.dictConfig(LOGGING)
logger = logging.getLogger(__name__)

_folder = os.path.expanduser('~')
if not os.access(_folder, os.W_OK):
    _folder = '/tmp'
_folder = os.path.join(_folder, '.art')

_config_path = os.path.expanduser(os.path.join(_folder, 'config.json'))
if os.path.exists(_config_path):
    try:
        with open(_config_path) as f:
            _config = json.load(f)
    except ValueError:
        _config = {}

if not os.path.exists(_folder):
    try:
        os.makedirs(_folder)
    except OSError:
        logger.warning('Unable to create folder for configuration file.', exc_info=True)

if not os.path.exists(_config_path):
    # Generate default config
    _config = {'DATA_PATH': os.path.join(_folder, 'data')}

    try:
        with open(_config_path, 'w') as f:
            f.write(json.dumps(_config, indent=4))
    except IOError:
        logger.warning('Unable to create configuration file', exc_info=True)

if 'DATA_PATH' in _config:
    DATA_PATH = _config['DATA_PATH']

NUMPY_DTYPE = float32


def get_date():
    # '20191007'
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def set_model_restore_path(restore_path):
    MODEL_RESTORE_PATH = restore_path
    # print(MODEL_RESTORE_PATH)