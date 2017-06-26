import sys, json, os, subprocess, pickle

import numpy as np
from scipy import io

from functools import reduce





""" Configuration
"""

if 'nipg' in os.uname().nodename:
    EXPERIMENTS_DIR = "/home/milacski/shota/USE_CASE_RNN_COMPLETION"
    PROGRAM = "/home/milacski/shota/fast-time-series-data-classification/algorithms/matrix_completion_rnn.py"
    TIME = "/usr/bin/time"
    IMPLEMENTATION = 2
elif os.uname().nodename == 'Regulus.local':
    EXPERIMENTS_DIR = "/Users/ngym/Lorincz-Lab/project/fast_time-series_data_classification/program/USE_CASE_RNN_COMPLETION"
    PROGRAM = "/Users/ngym/Lorincz-Lab/project/fast_time-series_data_classification/program/algorithms/matrix_completion_rnn.py"
    TIME = "gtime"
    IMPLEMENTATION = 1
elif os.uname().nodename.split('.')[0] in {'procyon', 'pollux', 'capella',
                                           'aldebaran', 'rigel'}:
    EXPERIMENTS_DIR = "/home/ngym/NFSshare/Lorincz_Lab/fast-time-series-data-classification/gak_gram_matrix_completion/USE_CASE_RNN_COMPLETION"
    PROGRAM = "/home/ngym/NFSshare/Lorincz_Lab/fast-time-series-data-classification/gak_gram_matrix_completion/matrix_completion_rnn.py"
    IMPLEMENTATION = 1
else:
    if len(sys.argv) == 3:
        EXPERIMENTS_DIR = sys.argv[1]
        IMPLEMENTATION = int(sys.argv[2])
        assert IMPLEMENTATION in {0, 1, 2}
    else:
        print("Specify an existing directory to build directies for experiments and" +\
              "\"implementation\" which is recommended to be 0 for CPU, 2 for GPU, 1 for any.")
        exit -1









""" k-fold cross-validation generators
"""

class KFold():
    """Base class for k-fold cross-validation test set generator.
    Puts ith sample into fold (i modulo fold count)
    I.e. each new sample goes to next fold
    Assumes that pkl['sample_names'] is sorted wrt classes

    :param pkl_file_path: .pkl file path for original Gram matrix
    :type pkl_file_path: str
    """
    
    def __init__(self, pkl_file_path):
        # assume pkl['sample_names'] is sorted with ground truth
        fd = open(pkl_file_path, 'rb')
        pkl = pickle.load(fd)
        self.sample_names = pkl['sample_names']
        self.generate_folds()
        
    def generate_folds(self):
        self.num_folds = 10
        self.fold = [[] for i in range(self.num_folds)]
        for i in range(len(self.sample_names)):
            self.fold[i % self.num_folds].append(i)

    def __iter__(self):
        self.k = 0
        return self

    def __next__(self):
        if self.k == len(self.fold):
            raise StopIteration()
        retval = self.fold[self.k]
        self.k += 1
        return retval    

class KFold_UCIauslan(KFold):
    """Class for k-fold cross-validation test set generator on UCI AUSLAN data set.
    This data set has 9 trials (recorded over 9 days) which defines a natural 9-fold separation.
 
    :param pkl_file_path: .pkl file path for original Gram matrix
    :type pkl_file_path: str
    """

    def __init__(self, pkl_file_path):
        super(KFold_UCIauslan, self).__init__(pkl_file_path)

    def generate_folds(self):
        self.num_folds = 9
        self.fold = [[] for i in range(self.num_folds)]
        for i in range(len(self.sample_names)):
            sample_name = self.sample_names[i]
            k = int(sample_name.split('-')[-2])
            self.fold[k - 1].append(i)








""" List of experiments to conduct.
"""

experiments = [
    {"dataset": "UCIauslan", "rnn": "LSTM", "units": [([10], [3])], "dropout": 0.3,
     "bidirectional": False, "batchnormalization": True},
    {"dataset": "UCIcharacter", "rnn": "LSTM", "units": [([10], [3])], "dropout": 0.3,
     "bidirectional": False, "batchnormalization": True},
    {"dataset": "6DMG", "rnn": "LSTM", "units": [([10], [3])], "dropout": 0.3,
     "bidirectional": False, "batchnormalization": True}
]










"""Creates directories, symbolic links to pkl files, k-fold cross-validation, json files, and timing for experiments.
"""

for exp in experiments:
    if exp['dataset'] is "UCIauslan":
        pkl_file_path = os.path.join(EXPERIMENTS_DIR, "original_gram_files/gram_UCIauslan_sigma12.000.pkl")
        sample_dir = os.path.join(EXPERIMENTS_DIR, "datasets/UCIauslan/all")
        kfold = KFold_UCIauslan
    elif exp['dataset'] is "UCIcharacter":
        pkl_file_path = os.path.join(EXPERIMENTS_DIR, "original_gram_files/gram_UCIcharacter_sigma20.000.pkl")
        sample_dir = os.path.join(EXPERIMENTS_DIR, "datasets/UCIcharacter")
        kfold = KFold
    elif exp['dataset'] is "6DMG":
        pkl_file_path = os.path.join(EXPERIMENTS_DIR, "original_gram_files/gram_upperChar_all_sigma20.000_t1-t3.pkl")
        sample_dir = os.path.join(EXPERIMENTS_DIR, "datasets/6DMG_mat_112712/matR_char")
        kfold = KFold
    else:
        raise ValueError("dataset must be one of UCIauslan, UCIcharacter or 6DMG")

    if exp['bidirectional']:
        direc = os.path.join(exp['dataset'], exp['rnn'], "Bidirectional")
    else:
        direc = os.path.join(exp['dataset'], exp['rnn'], "Forward")

    if exp['batchnormalization']:
        direc = os.path.join(direc, "BatchNormalization")
    else:
        direc = os.path.join(direc, "NoBatchNormalization")
        
    for rnn_units, dense_units in exp['units']:
        folds = kfold(pkl_file_path)
        k = 0
        for fold in folds:
            k_dir = os.path.join(EXPERIMENTS_DIR,
                                 direc,
                                 str(rnn_units),
                                 str(dense_units),
                                 str(exp['dropout']),
                                 str(k))

            try:
                os.makedirs(k_dir)
            except FileExistsError:
                pass

            pkl_file = pkl_file_path.split("/")[-1]

            subprocess.run(["ln", "-s", pkl_file_path, k_dir])
            gram_file = os.path.join(k_dir, pkl_file)
            completionanalysisfile = gram_file.replace(".pkl", ".timelog")

            json_dict = dict(gram_file=gram_file,
                             sample_dir=sample_dir,
                             indices_to_drop=fold,
                             completionanalysisfile=os.path.join(k_dir,
                                                                 completionanalysisfile),
                             epochs=100,
                             patience=2,
                             dataset=exp['dataset'],
                             rnn=exp['rnn'],
                             rnn_units=rnn_units,
                             dense_units=dense_units,
                             dropout=exp['dropout'],
                             implementation=IMPLEMENTATION,
                             bidirectional=exp['bidirectional'],
                             batchnormalization=exp['batchnormalization'])

            json_file_name = os.path.join(k_dir, "config_rnn_conpletion.json")
            fd = open(json_file_name, "w")
            json.dump(json_dict, fd)
            fd.close()

            if os.uname().nodename.split('.')[0] in {'procyon', 'pollux', 'capella',
                                                     'aldebaran', 'rigel'}:
                job_file_name = os.path.join(k_dir, pkl_file + "_k" + str(k) + ".job")
                fd = open(job_file_name, "w")
                time_file_name = os.path.join(k_dir, "time_command.output")

                fd.write("echo $SHELL\n")
                fd.write("setenv LD_LIBRARY_PATH /home/ngym/NFSshare/tflib/lib64/:/home/ngym/NFSshare/tflib/usr/lib64/\n")
                fd.write("~/NFSshare/tflib/lib64/ld-2.17.so /usr/bin/time -v -o " +\
                         time_file_name + \
                         " ~/NFSshare/tflib/lib64/ld-2.17.so /usr/bin/python3 " +\
                         PROGRAM + " " + json_file_name + "\n")
                fd.close()
            else:
                command_file_name = os.path.join(k_dir, "command.sh")
                time_file_name = os.path.join(k_dir, "time_command.output")
                fd = open(command_file_name, "w")
                fd.write(TIME + " -v -o " + time_file_name +\
                         " python3 " +\
                         PROGRAM + " " + json_file_name + "\n")
                fd.close()

            k += 1

