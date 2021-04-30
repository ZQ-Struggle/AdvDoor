"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import pickle

import numpy as np
import tensorflow as tf
from tqdm import trange
import keras.backend as K
from utils import *


def compute_corr(param):
    # seeding randomness
    with open(param.get_conf('model_path_backdoor'), 'rb') as f:
        model = pickle.load(f)
    model.set_learning_phase(0)
    if param.get_conf('model_prefix') == 'mnist':
        from data.mnist import MnistData
        data = MnistData(param)
        iteation = K.function(inputs=[model.get_classifier().get_model().input],
                              outputs=[model.get_classifier().get_model().get_layer('dense_1').output])
    elif param.get_conf('model_prefix') == 'cifar':
        from data.cifar10 import CifarData
        data = CifarData(param)
        iteation = K.function(inputs=[model.get_classifier().get_model().input],
                              outputs=[model.get_classifier().get_model().get_layer('dense_2').output])
    data.load_data()
    data.restore_backdoor(model)
    # Setting up the data and the model
    train_x, train_y, test_x, test_y, is_poison_train, is_poison_test = data.get_specific_label_data(6)
    target_label = param.get_conf('poison_label_target')
    num_poisoned_left = np.sum(is_poison_train == True)
    print('Num poisoned left: ', num_poisoned_left)
    num_training_examples = len(train_x)


    print('Dataset Size: ', len(data.x_train))

    lbl = target_label
    cur_examples = num_training_examples
    print('Label, num ex: ', lbl, cur_examples)
    # cur_op = model.representation
    for iex in trange(cur_examples):
        x_batch = train_x[iex:iex + 1, :]
        y_batch = train_y[iex:iex + 1]

        batch_grads = iteation([x_batch])[0].flatten()

        if iex == 0:
            clean_cov = np.zeros(shape=(cur_examples - num_poisoned_left, len(batch_grads)))
            full_cov = np.zeros(shape=(cur_examples, len(batch_grads)))
        if iex < (cur_examples - num_poisoned_left):
            clean_cov[iex] = batch_grads
        full_cov[iex] = batch_grads

    # np.save(corr_dir+str(lbl)+'_full_cov.npy', full_cov)
    
    total_p = 73


    clean_mean = np.mean(clean_cov, axis=0, keepdims=True)
    full_mean = np.mean(full_cov, axis=0, keepdims=True)

    print('Norm of Difference in Mean: ', np.linalg.norm(clean_mean - full_mean))
    clean_centered_cov = clean_cov - clean_mean
    s_clean = np.linalg.svd(clean_centered_cov, full_matrices=False, compute_uv=False)
    print('Top 7 Clean SVs: ', s_clean[0:7])

    centered_cov = full_cov - full_mean
    u, s, v = np.linalg.svd(centered_cov, full_matrices=False)
    print('Top 7 Singular Values: ', s[0:7])
    eigs = v[0:1]
    p = total_p
    corrs = np.matmul(eigs, np.transpose(full_cov))  # shape num_top, num_active_indices
    scores = np.linalg.norm(corrs, axis=0)  # shape num_active_indices
    # np.save(os.path.join(model_dir, 'scores.npy'), scores)
    print('Length Scores: ', len(scores))
    p_score = np.percentile(scores, p)
    top_scores = np.where(scores > p_score)[0]
    print(top_scores)

    num_bad_removed = np.sum(is_poison_train[top_scores])
    print('Num Bad Removed: ', num_bad_removed)
    print('Num Good Rmoved: ', len(top_scores) - num_bad_removed)

    num_poisoned_after = num_poisoned_left - num_bad_removed

    print('Num Poisoned Left: ', num_poisoned_after)

    print_f1(num_bad_removed, num_poisoned_after, len(top_scores) - num_bad_removed)

    if os.path.exists('job_result.json'):
        with open('job_result.json') as result_file:
            result = json.load(result_file)
            result['num_poisoned_left'] = '{}'.format(num_poisoned_after)
    else:
        result = {'num_poisoned_left': '{}'.format(num_poisoned_after)}
    with open('job_result.json', 'w') as result_file:
        json.dump(result, result_file, sort_keys=True, indent=4)


