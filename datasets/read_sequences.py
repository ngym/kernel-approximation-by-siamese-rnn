import os, csv
import os.path as path
from collections import OrderedDict
import glob

import numpy as np
from scipy import io

from datasets import others


def read_sequences(dataset_type, list_glob_arg=None, direc=None,
                   feature_normalization=False,
                   data_attribute_type=None):
    """Time series loader.
    Parse time series from files.
    UCIauslan is assumed that the contents of subdirectories tctodd1-tctodd9
    is merged in a directory and tctodd1/hers-1.tsd, tctodd1/hers-2.tsd and
    tctodd1/hers-3.tsd are fixed to his_hers-[1-3].tsd.

    :param dataset_type: 6DMGupperChar or UCIauslan or UCIcharacter
    :param list_glob_arg: List of, argument to pass to glob.glob which holds sequence data.
    :param direc: Directory which holds dataset
    :type dataset_type: str
    :type glob_arg: str
    :returns: dict of list of time series
    :rtype: dict of np.ndarrays
    """

    if list_glob_arg is None and direc is None\
       or list_glob_arg is not None and direc is not None:
        print("Either list_glob_arg or direc is required.")
        return -1

    sample_files = []
    if list_glob_arg is not None:
        for glob_arg in list_glob_arg:
            sample_files += glob.glob(glob_arg)
    elif direc is not None:
        if dataset_type in {"UCIauslan", "UCItctodd"}:
            extension = "*.tsd"
        elif dataset_type in {"6DMG", "6DMGupperChar", "upperChar"}:
            extension = "upper_*.mat"
        else:
            extension = "*.mat"
        sample_files = glob.glob(path.join(direc, extension))
    sample_files.sort(key=lambda fn: fn.split('/')[-1])
    
    seqs = OrderedDict()
    if dataset_type in {"6DMG", "6DMGupperChar", "upperChar"}:
        for sample_file in sample_files:
            m = io.loadmat(sample_file)
            if data_attribute_type is None or data_attribute_type == "all":
                seqs[others.get_sample_name(dataset_type, sample_file)]\
                    = m['gest'].T[:, 1:]
            elif data_attribute_type == "position":
                seqs[others.get_sample_name(dataset_type, sample_file)]\
                    = m['gest'].T[:, 1:4]
            elif data_attribute_type == "velocity":
                seqs[others.get_sample_name(dataset_type, sample_file)]\
                    = (m['gest'].T[1:, 1:4] - m['gest'].T[:-1, 1:4]) / 1000
            elif data_attribute_type == "orientation":
                seqs[others.get_sample_name(dataset_type, sample_file)]\
                    = m['gest'].T[:, 4:8]
            elif data_attribute_type == "acceleration":
                seqs[others.get_sample_name(dataset_type, sample_file)]\
                    = m['gest'].T[:, 8:11]
            elif data_attribute_type == "angularvelocity":
                seqs[others.get_sample_name(dataset_type, sample_file)]\
                = m['gest'].T[:, 11:14]
            else:
                print("attribute type error.")
                assert False
    elif dataset_type == "UCIcharacter":
        if isinstance(list_glob_arg, str):
            mat_file_path = list_glob_arg
        else:
            mat_file_path = sample_files[0]
        data = io.loadmat(mat_file_path)
        displayname = [k[0] for k in data['consts']['key'][0][0][0]]
        classes = data['consts'][0][0][4][0]
        labels = []
        for c in classes:
            labels.append(displayname[c-1])
        i = 0
        seqs_ = []
        for l in labels:
            seqs_.append((l + str(i), data['mixout'][0][i].T))
            i += 1
        for k, v in sorted(seqs_):
            seqs[others.get_sample_name(dataset_type, k)] = v
    elif dataset_type == "UCIauslan":
        for sample_file in sample_files:
            reader = csv.reader(open(sample_file, "r"),
                                delimiter='\t')
            seq = []
            for r in reader:
                seq.append(r)
            seqs[others.get_sample_name(dataset_type, sample_file)] = np.array(seq).astype(np.float32)
    elif dataset_type == "UCIarabicdigits":
        def add_seq(prefix, i, seqs, seq):
            if prefix == "test":
                label = i // 220
                num = i % 220
            else:
                label = i // 660
                num = i % 660
            name = prefix + "_" + str(label) + "_" + str(num)
            seqs[name] = np.array(seq).astype(np.float32)
        # space separated, blank line separated
        filenames = {("test", os.path.join(direc, "Test_Arabic_Digit.txt")),
                     ("train", os.path.join(direc, "Train_Arabic_Digit.txt"))}
        for prefix, filename in filenames:
            with open(filename, "r") as fd:
                reader = csv.reader(fd, delimiter=' ')
                seq = []
                i = 0
                next(reader)
                for r in reader:
                    if r == ['', '', '', '', '', '', '', '', '', '', '', '', '']:
                        add_seq(prefix, i, seqs, seq)
                        seq = []
                    else:
                        seq.append(r)
                    i += 1
                add_seq(prefix, i, seqs, seq)
    else:
        assert False
    key_to_str, key_to_int = get_labels(seqs, dataset_type)

    if feature_normalization:
        seqs = normalization(seqs)
    
    return seqs, key_to_str, key_to_int


def get_labels(seqs, dataset_type):
    """Get labels from sequences.
    :param seqs: A dictionary of list of time series.
    :param dataset_type: 6DMGupperChar or UCIauslan or UCIcharacter.
    :type seqs: collections.OrderedDict
    :type dataset_type: str
    :return: A dictionary pair which are represents the labels of the given sequences.
    :rtype: (collections.OrderedDict, collections.OrderedDict)
    """
    if not others.is_valid_dataset_type(dataset_type):
        assert False

    key_to_str = OrderedDict()
    key_to_int = OrderedDict()
    label_to_int = dict()
    last_index = 0
    for k, _ in seqs.items():
        label = others.get_label(dataset_type, k)
        if label not in label_to_int:
            label_to_int[label] = last_index
            last_index += 1
        key_to_str[k] = label
        key_to_int[k] = label_to_int[label]

    return key_to_str, key_to_int

def normalization(seqs):
    newseqs = OrderedDict()
    mean = np.mean(np.concatenate(np.array(list(seqs.values())), axis=0),
                   axis=0, keepdims=True)
    std = np.std(np.concatenate(np.array(list(seqs.values())), axis=0),
                   axis=0, keepdims=True)
    for k, v in seqs.items():
        newseqs[k] = (v - mean) / (std + 1e-8)
    return newseqs

def pick_labels(dataset_type, seqs, labels_to_use):
    new_seq = OrderedDict()
    for k, v in seqs.items():
        label = others.get_label(dataset_type, k)
        if label in labels_to_use:
            new_seq[k] = v
    return new_seq






