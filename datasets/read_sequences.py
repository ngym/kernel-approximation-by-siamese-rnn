import os, csv
import os.path as path
from collections import OrderedDict
import glob

import numpy as np
from scipy import io

from datasets import others


def read_sequences(dataset_type, 
                   dataset_location,
                   feature_normalization=False,
                   data_attribute_type=None):
    """Time series loader.
    Parse time series from files.

    :param dataset_type: 6DMGupperChar or UCIauslan or UCIcharacter
    :param dataset_location: Directory which holds dataset
    :data_attribute_type: Attribute to use. For 6DMG.

    :type dataset_type: str
    :type dataset_location: str
    :type data_attribute_type: str
    :returns: dict of list of time series
    :rtype: dict of np.ndarrays
    """
    
    seqs = []
    names = []
    if dataset_type in {"6DMG", "6DMGupperChar", "upperChar"}:
        sample_files = glob.glob(path.join(dataset_location, "upper_*.mat"))
        sample_files.sort(key=lambda fn: fn.split('/')[-1])
        for sample_file in sample_files:
            m = io.loadmat(sample_file)
            if data_attribute_type is None or data_attribute_type == "all":
                seqs.append(m['gest'].T[:, 1:])
            elif data_attribute_type == "position":
                seqs.append(m['gest'].T[:, 1:4])
            elif data_attribute_type == "velocity":
                seqs.append((m['gest'].T[1:, 1:4] - m['gest'].T[:-1, 1:4]))
            elif data_attribute_type == "orientation":
                seqs.append(m['gest'].T[:, 4:8])
            elif data_attribute_type == "acceleration":
                seqs.append(m['gest'].T[:, 8:11])
            elif data_attribute_type == "angularvelocity":
                seqs.append(m['gest'].T[:, 11:14])
            else:
                print("attribute type error.")
                assert False
            names.append(others.get_sample_name(dataset_type, sample_file))
    elif dataset_type == "UCIcharacter":
        data = io.loadmat(dataset_location)
        displayname = [k[0] for k in data['consts']['key'][0][0][0]]
        classes = data['consts'][0][0][4][0]
        names_ = []
        for c in classes:
            names_.append(displayname[c-1])
        i = 0
        seqs_ = []
        for l in names_:
            seqs_.append((l + str(i), data['mixout'][0][i].T))
            i += 1
        for k, v in sorted(seqs_):
            seqs.append(v)
            names.append(others.get_sample_name(dataset_type, k))
    elif dataset_type == "UCIauslan":
        sample_files = glob.glob(path.join(dataset_location, "*/*.tsd"))
        sample_files.sort(key=lambda fn: fn.split('/')[-1])
        for sample_file in sample_files:
            reader = csv.reader(open(sample_file, "r"),
                                delimiter='\t')
            seq = []
            for r in reader:
                seq.append(r)
            seqs.append(np.array(seq).astype(np.float32))
            names.append(others.get_sample_name(dataset_type, sample_file))
    elif dataset_type == "UCIarabicdigits":
        def add_seq(prefix, i, seqs, seq):
            if prefix == "test":
                label = i // 220
                num = i % 220
            else:
                label = i // 660
                num = i % 660
            name = prefix + "_" + str(label) + "_" + str(num)
            seqs.append(np.array(seq).astype(np.float32))
            names.append(name)
        # space separated, blank line separated
        filenames = {("test", os.path.join(dataset_location, "Test_Arabic_Digit.txt")),
                     ("train", os.path.join(dataset_location, "Train_Arabic_Digit.txt"))}
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
                        i += 1
                    else:
                        seq.append(r)
                add_seq(prefix, i, seqs, seq)
    else:
        assert False
    labels_str, labels_int = get_labels(names, dataset_type)

    if feature_normalization:
        seqs = normalization(seqs)
    
    return seqs, names, labels_str, labels_int


def get_labels(names, dataset_type):
    """Get labels from sequences.
    :param names: A list of names of time series.
    :param dataset_type: 6DMGupperChar or UCIauslan or UCIcharacter.
    :type names: list
    :type dataset_type: str
    :return: labels of the given sequences.
    :rtype: (list, list)
    """
    if not others.is_valid_dataset_type(dataset_type):
        assert False

    key_to_str = []
    key_to_int = []
    label_to_int = dict()
    last_index = 0
    for name in names:
        label = others.get_label(dataset_type, name)
        if label not in label_to_int:
            label_to_int[label] = last_index
            last_index += 1
        key_to_str.append(label)
        key_to_int.append(label_to_int[label])

    return key_to_str, key_to_int

def normalization(seqs):
    new_seqs = []
    mean = np.mean(np.concatenate(seqs, axis=0),
                   axis=0, keepdims=True)
    std = np.std(np.concatenate(seqs, axis=0),
                 axis=0, keepdims=True)
    for s in seqs:
        new_seqs.append((s - mean) / (std + 1e-8))
    return new_seqs

def pick_labels(dataset_type, seqs, names, labels_to_use):
    new_seqs = []
    new_names = []
    for n in range(len(seqs)):
        s = seqs[n]
        n = names[n]
        label = others.get_label(dataset_type, n)
        if label in labels_to_use:
            new_seqs.append(s)
            new_names.append(n)
    return new_seq, new_names






