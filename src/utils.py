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


def init_weights(rules):
    return np.random.uniform(-0.005, 0.005, len(rules))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def write_all_triplet(save_path):
    return

