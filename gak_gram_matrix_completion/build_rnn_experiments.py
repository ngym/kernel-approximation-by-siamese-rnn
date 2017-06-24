import sys, json, os, subprocess

import numpy as np
from scipy import io

from functools import reduce





""" Configuration
"""

if 'nipg' in os.uname().nodename:
    EXPERIMENTS_DIR = "~/shota/USE_CASE_RNN_COMPLETION"
    PROGRAM = "~/shota/fast-time-series-data-classification/gak_gram_matrix_completion/matrix_completion_rnn.py"
    TIME = "/usr/bin/time"
    IMPLEMENTATION = 2
elif os.uname().nodename == 'Regulus.local':
    EXPERIMENTS_DIR = "/Users/ngym/Lorincz-Lab/project/fast_time-series_data_classification/program/gak_gram_matrix_completion/USE_CASE_RNN_COMPLETION"
    PROGRAM = "/Users/ngym/Lorincz-Lab/project/fast_time-series_data_classification/program/gak_gram_matrix_completion/matrix_completion_rnn.py"
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
    Assumes that mat['indices'] is sorted wrt classes

    :param mat_file_path: .mat file path for original Gram matrix
    :type mat_file_path: str
    """
    
    def __init__(self, mat_file_path):
        # assume mat['indices'] is sorted with ground truth
        mat = io.loadmat(mat_file_path)
        indices = mat['indices']
        self.generate_folds()
        
    def generate_folds():
        self.num_folds = 10
        self.fold = [[] for i in range(self.num_folds)]
        for i in range(len(indices)):
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
 
    :param mat_file_path: .mat file path for original Gram matrix
    :type mat_file_path: str
    """

    def __init__(self, mat_file_path):
        super(KFold_UCIauslan, self).__init__(mat_file_path)

    def generate_folds():
        self.num_folds = 9
        self.fold = [[] for i in range(self.num_folds)]
        for i in range(len(indices)):
            index = indices[i]
            index_ = index.split('/')[-1]
            k = int(index_.split('-')[-2])
            self.fold[k - 1].append(i)








""" List of experiments to conduct.
"""

experiments = [
    {"dataset": "UCIauslan", "rnn": "LSTM", "units": [([10], [3])], "dropout": 0.3, "bidirectional": False},
    {"dataset": "UCIcharacter", "rnn": "LSTM", "units": [([10], [3])], "dropout": 0.3, "bidirectional": False},
    {"dataset": "6DMG", "rnn": "LSTM", "units": [([10], [3])], "dropout": 0.3, "bidirectional": False}]
]










"""This part of the code creates directories, symbolic links to mat files, k-fold cross-validation, json files, and timing for experiments.
"""

for exp in experiments:
    if exp['dataset'] is "UCIauslan":
        mat_file_path = os.path.join(EXPERIMENTS_DIR, "original_gram_files/gram_UCIauslan_sigma12.000.mat"),
        kfold = KFold_UCIauslan
    elif exp['dataset'] is "UCIcharacter":
        mat_file_path = os.path.join(EXPERIMENTS_DIR, "original_gram_files/gram_UCIcharacter_sigma20.000.mat"),
        kfold = KFold
    elif exp['dataset'] is "6DMG":
        mat_file_path = os.path.join(EXPERIMENTS_DIR, "original_gram_files/gram_upperChar_all_sigma20.000_t1-t3.mat"),
        kfold = KFold
    else:
        raise ValueError("dataset must be one of UCIauslan, UCIcharacter or 6DMG")

    if exp['bidirectional']:
        direc = os.path.join(exp['dataset'], exp['rnn'], "Bidirectional")
    else:
        direc = os.path.join(exp['dataset'], exp['rnn'], "Forward")
    for lstm_units, dense_units in exp['units']:
        folds = kfold(mat_file_path)
        k = 0
        for fold in folds:
            k_dir = os.path.join(EXPERIMENTS_DIR,
                                 direc,
                                 str(lstm_units),
                                 str(dense_units),
                                 str(exp['dropout']),
                                 str(k))

            try:
                os.makedirs(k_dir)
            except FileExistsError:
                pass

            mat_file = mat_file_path.split("/")[-1]

            subprocess.run(["ln", "-s", mat_file_path, k_dir])
            gram_file = os.path.join(k_dir, mat_file)
            completionanalysisfile = gram_file.replace(".mat", ".timelog")

            json_dict = dict(gram_file=gram_file,
                             indices_to_drop=fold,
                             completionanalysisfile=os.path.join(k_dir,
                                                                 completionanalysisfile),
                             epochs=100,
                             patience=2,
                             dataset=exp['dataset'],
                             rnn=exp['rnn'],
                             lstm_units=lstm_units,
                             dense_units=dense_units,
                             dropout=exp['dropout'],
                             implementation=IMPLEMENTATION,
                             bidirectional=exp['bidirectional'])

            json_file_name = os.path.join(k_dir, "config_rnn_conpletion.json")
            fd = open(json_file_name, "w")
            json.dump(json_dict, fd)
            fd.close()

            if os.uname().nodename.split('.')[0] in {'procyon', 'pollux', 'capella',
                                                     'aldebaran', 'rigel'}:
                job_file_name = os.path.join(k_dir, mat_file + "_k" + str(k) + ".job")
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

