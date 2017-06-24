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

def find_and_read_sequences(filename, files):
    """Data set loader.
    Parse time series from files
    TODO: automatic files from data set name
    
    :param filename: Data set name
    :param files: List of files to parse
    :type filename: str
    :type files: list of str
    :returns: List of time series
    :rtype: list of np.ndarray
    """
    
    seqs = OrderedDict()
    if filename.find("upperChar") != -1 or filename.find("velocity") != -1:
        for f in files:
            #print(f)
            if 'nipg' in os.uname().nodename:
                m = io.loadmat(f.replace("/home/ngym/NFSshare/Lorincz_Lab",
                                         "~/shota/dataset"))
            elif os.uname().nodename == 'atlasz' or 'cn' in os.uname().nodename:
                m = io.loadmat(f.replace("/home/ngym/NFSshare/Lorincz_Lab",
                                         "/users/milacski/shota/dataset"))
            elif os.uname().nodename == 'Regulus.local':
                m = io.loadmat(f.replace("/home/ngym/NFSshare/Lorincz_Lab",
                                         "/Users/ngym/Lorincz-Lab/project/fast_time-series_data_classification/dataset"))
            else:
                m = io.loadmat(f)
            seqs[f] = m['gest'].T
    elif filename.find("UCIcharacter") != -1:
        if 'nipg' in os.uname().nodename:
            datasetfile = "~/shota/dataset/mixoutALL_shifted.mat"
        elif os.uname().nodename == 'atlasz' or 'cn' in os.uname().nodename:
            datasetfile = "/users/milacski/shota/dataset/mixoutALL_shifted.mat"
        elif os.uname().nodename == 'Regulus.local':
            datasetfile = "/Users/ngym/Lorincz-Lab/project/fast_time-series_data_classification/dataset/UCI/mixoutALL_shifted.mat"
        else:
            datasetfile = "/home/ngym/NFSshare/Lorincz_Lab/mixoutALL_shifted.mat"
        dataset = io.loadmat(datasetfile)
        displayname = [k[0] for k in dataset['consts']['key'][0][0][0]]
        classes = dataset['consts'][0][0][4][0]
        labels = []
        for c in classes:
            labels.append(displayname[c-1])
        i = 0
        for l in labels:
            seqs[l + str(i)] = dataset['mixout'][0][i].T
            i += 1
    elif filename.find("UCIauslan") != -1:
        for f in files:
            if 'nipg' in os.uname().nodename:
                reader = csv.reader(open(f.replace(' ', '')\
                                         .replace("/home/ngym/NFSshare/Lorincz_Lab",
                                                  "~/shota/dataset"),
                                    "r"), delimiter='\t')
            elif os.uname().nodename == 'atlasz' or 'cn' in os.uname().nodename:
                reader = csv.reader(open(f.replace(' ', '')\
                                         .replace("/home/ngym/NFSshare/Lorincz_Lab",
                                                  "/users/milacski/shota/dataset"),
                                     "r"), delimiter='\t')
            elif os.uname().nodename == 'Regulus.local':
                reader = csv.reader(open(f.replace(' ', '')\
                                         .replace("/home/ngym/NFSshare/Lorincz_Lab",
                                                  "/Users/ngym/Lorincz-Lab/project/fast_time-series_data_classification/dataset")\
                                         .replace("/UCI", "/UCI/AUSLAN"),
                                    "r"), delimiter='\t')
            else:
                reader = csv.reader(open(f.replace(' ', ''), "r"), delimiter='\t')
            seq = []
            for r in reader:
                seq.append(r)
            seqs[f] = np.array(seq).astype(np.float32)
    else:
        assert False
    return seqs

