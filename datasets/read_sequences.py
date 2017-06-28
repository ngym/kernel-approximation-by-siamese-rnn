import os, csv
import os.path as path
from collections import OrderedDict
import glob

import numpy as np
from scipy import io

def read_sequences(dataset_type, list_glob_arg=None, direc=None):
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

    if list_glob_arg is not None:
        sample_files = []
        for glob_arg in list_glob_arg:
            sample_files_ = glob.glob(glob_arg)
            sample_files += sample_files_
        sample_files = sorted(sample_files)
    elif direc is not None:
        if dataset_type in {"UCIauslan", "UCItctodd"}:
            extension = "*.tsd"
        else:
            extension = "*.mat"
        sample_files = sorted(glob.glob(path.join(direc, extension)))
    
    seqs = OrderedDict()
    if dataset_type in {"6DMG", "6DMGupperChar", "upperChar"}:
        for sample_file in sample_files:
            m = io.loadmat(sample_file)
            seqs[sample_file] = m['gest'].T[:, 1:]
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
            seqs[k] = v
    elif dataset_type == "UCIauslan":
        for sample_file in sample_files:
            reader = csv.reader(open(sample_file.replace(' ', ''), "r"),
                                delimiter='\t')
            seq = []
            for r in reader:
                seq.append(r)
            seqs[sample_file] = np.array(seq).astype(np.float32)
    else:
        assert False
    return seqs

