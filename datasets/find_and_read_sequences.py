import sys, random, copy, os, gc, time, csv, json
import os.path as path
from collections import OrderedDict
from tempfile import mkdtemp

import plotly.offline as po
import plotly.graph_objs as pgo

import numpy as np
import scipy as sp
from scipy import io
from scipy.io import wavfile
from scipy import signal

def read_sequences(dataset_type, direc, sample_names, attribute_type=None):
    """Time series loader.
    Parse time series from files

    :param dataset_type: 6DMGupperChar or UCIauslan or UCIcharacter
    :param direc: directory which holds sequence data
    :param sample_names: List of data files to parse
    :param attribute_type: velocity, position, etc. of 6DMG
    :type dataset_type: str
    :type direc: str
    :type sample_names: str for UCIcharacter or list of str for others
    :type attribute_type: str
    :returns: dictionary of list of time series
    :rtype: dictionary of np.ndarray
    """
    
    seqs = OrderedDict()
    if dataset_type in {"6DMG", "6DMGupperChar", "upperChar"}:
        for sample_name in sample_names:
            f = path.join(direc, sample_name)
            m = io.loadmat(f)
            seqs[f] = m['gest'].T
    elif dataset_type == "UCIcharacter":
        mat_file = "mixoutALL_shifted.mat"
        mat_file_path = path.join(direc, mat_file)
        data = io.loadmat(mat_file_path)
        displayname = [k[0] for k in data['consts']['key'][0][0][0]]
        classes = data['consts'][0][0][4][0]
        labels = []
        for c in classes:
            labels.append(displayname[c-1])
        i = 0
        for l in labels:
            seqs[l + str(i)] = data['mixout'][0][i].T
            i += 1
    elif dataset_type == "UCIauslan":
        for sample_name in sample_names:
            f = path.join(direc, sample_name)
            reader = csv.reader(open(f.replace(' ', ''), "r"), delimiter='\t')
            seq = []
            for r in reader:
                seq.append(r)
            seqs[f] = np.array(seq).astype(np.float32)
    else:
        assert False
    return seqs

