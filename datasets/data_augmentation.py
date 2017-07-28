import os, csv
import os.path as path
from collections import OrderedDict
import glob

import numpy as np
from scipy import io
from scipy.stats import truncnorm

from datasets import others

from datasets.read_sequences import read_sequences

def data_augmentation(dataset_type, list_glob_arg=None, direc=None,
                      feature_normalization=False,
                      data_attribute_type=None):
    seqs = read_sequences(dataset_type, list_glob_arg=list_glob_arg, direc=direc,
                          feature_normalization=feature_normalization,
                          data_attribute_type=data_attribute_type)

def augment_random(seq, length):
    while seq.shape[0] < length:
        seq = seq_insert(seq, np.random.randint(seq.shape[0] - 1))
    while seq.shape[0] > length:
        seq = seq_delete(seq, np.random.randint(seq.shape[0] - 1))
    return seq

def augment_normal_distribution(seq, length, ave_, std_):
    ave = ave_
    std = std_
    std_fraction = std / len(seq)
    while seq.shape[0] < length:
        std = len(seq) * std_fraction
        time_insert = int(truncnorm.rvs((0 - ave) / std,
                                    (seq.shape[0] - 1 - ave) / std,
                                    loc=ave,
                                    scale=std))
        seq = seq_insert(seq, time_insert)
        if time_insert > ave:
            ave += 1
    while seq.shape[0] > length:
        time_remove = int(truncnorm.rvs((0 - ave) / std,
                                    (seq.shape[0] - 1 - ave) / std,
                                    loc=ave,
                                    scale=std))
        seq = seq_delete(seq, time_remove)
        if time_remove > ave:
            ave += 1
    return seq

def seq_insert(seq, time_insert):
    new_step = (seq[time_insert] + seq[time_insert + 1]) / 2
    seq = np.insert(seq, time_insert + 1, new_step, axis=0)
    return seq

def seq_delete(seq, time_delete):
    new_seq = np.delete(seq, time_delete, axis=0)
    return new_seq



