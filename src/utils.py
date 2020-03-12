from os.path import dirname, abspath, exists, join
from os import makedirs
import numpy as np
import math


def get_root_path():
    return dirname(dirname(abspath(__file__)))


def get_data_path():
    return join(get_root_path(), 'data')


def get_save_path():
    return join(get_root_path(), 'save')


def ensure_dir(d):
    if not exists(d):
        makedirs(d)


def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def init_weights(rules_to_w_idx):
    w_numpy = np.zeros(len(rules_to_w_idx))

    for rule, w_idx in rules_to_w_idx.items():
        if rule == 'notHidden':
            w_numpy[w_idx] = np.random.normal(10, 0.05)
        else:
            w_numpy[w_idx] = np.random.normal(1, 0.05)
    return w_numpy


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def shifted_sigmoid(x):
    return 1 / (1 + math.exp(-(x-0.3)))


def write_all_triplet(save_path):
    return

