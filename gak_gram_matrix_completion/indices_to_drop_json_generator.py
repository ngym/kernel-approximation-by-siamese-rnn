import sys, json, os, subprocess

import numpy as np
from scipy import io

from functools import reduce

if os.uname().nodename == 'Regulus.local':
    USE_CASE_RNN_COMPLETION_DIR = "/Users/ngym/Lorincz-Lab/project/fast_time-series_data_classification/program/gak_gram_matrix_completion/USE_CASE_RNN_COMPLETION"
elif os.uname().nodename == 'nipgcore1':
    USE_CASE_RNN_COMPLETION_DIR = "/home/milacski/shota/USE_CASE_RNN_COMPLETION"
elif os.uname().nodename.split('.')[0] in {'procyon', 'pollux', 'capella',
                                           'aldebaran', 'rigel'}:
    USE_CASE_RNN_COMPLETION_DIR = "/home/ngym/NFSshare/Lorincz_Lab/fast-time-series-data-classification/gak_gram_matrix_completion/USE_CASE_RNN_COMPLETION"
    PROGRAM = "/home/ngym/NFSshare/Lorincz_Lab/fast-time-series-data-classification/gak_gram_matrix_completion/matrix_completion_rnn_residual.py"
else:
    print("unsupported server")
    exit -1

class Drop_generator_UCItctodd():
    def __init__(self, orig_gram_file_path):
        # k_group is divided by the date UCI AUSLAN is observed.
        mat = io.loadmat(orig_gram_file_path)
        indices = mat['indices']

        num_groups = 9
        self.__groups = [[] for i in range(num_groups)]
        for i in range(len(indices)):
            index = indices[i]
            index_ = index.split('/')[-1]
            k_group = int(index_.split('-')[-2])
            ground_truth = reduce(lambda a, b: a + "-" + b, index_.split('-')[:-2])
            self.__groups[k_group-1].append(i)
    def __iter__(self):
        self.__i = 0
        return self
    def __next__(self):
        if self.__i == len(self.__groups):
            raise StopIteration()
        retval = self.__groups[self.__i]
        self.__i += 1
        return retval

class Drop_generator_UCIcharacter():
    def __init__(self, orig_gram_file_path):
        mat = io.loadmat(orig_gram_file_path)
        indices = mat['indices']

        num_groups = 10
        self.__groups = [[] for i in range(num_groups)]
        for i in range(len(indices)):
            self.__groups[i % num_groups].append(i)
    def __iter__(self):
        self.__i = 0
        return self
    def __next__(self):
        if self.__i == len(self.__groups):
            raise StopIteration()
        retval = self.__groups[self.__i]
        self.__i += 1
        return retval

class Drop_generator_6DMG():
    def __init__(self, orig_gram_file_path):
        # assume mat['indices'] is sorted with ground truth
        k_groups = ["A1", "C1", "C2", "C3", "C4", "E1", "G1", "G2", "G3", "I1",
                    "I2", "I3", "J1", "J2", "J3", "L1", "M1", "S1", "T1", "U1",
                    "Y1", "Y2", "Y3", "Z1", "Z2"]
        mat = io.loadmat(orig_gram_file_path)
        indices = mat['indices']
        
        num_groups = 10
        self.__groups = [[] for i in range(num_groups)]
        for i in range(len(indices)):
            self.__groups[i % num_groups].append(i)
    def __iter__(self):
        self.__i = 0
        return self
    def __next__(self):
        if self.__i == len(self.__groups):
            raise StopIteration()
        retval = self.__groups[self.__i]
        self.__i += 1
        return retval

dataset_settings = [
    ("UCItctodd/RNN", 5, 2,
     os.path.join(USE_CASE_RNN_COMPLETION_DIR,
                  "original_gram_files/gram_UCItctodd_sigma12.000.mat"),
     Drop_generator_UCItctodd(
         os.path.join(USE_CASE_RNN_COMPLETION_DIR,
                      "original_gram_files/gram_UCItctodd_sigma12.000.mat")
     )),
    ("UCIcharacter/RNN", 5, 2,
     os.path.join(USE_CASE_RNN_COMPLETION_DIR,
                  "original_gram_files/gram_UCIcharacter_sigma20.000.mat"),
     Drop_generator_UCIcharacter(
         os.path.join(USE_CASE_RNN_COMPLETION_DIR,
                      "original_gram_files/gram_UCIcharacter_sigma20.000.mat"))),
    ("6DMG/RNN", 5, 2,
     os.path.join(USE_CASE_RNN_COMPLETION_DIR,
                  "original_gram_files/gram_upperChar_all_sigma20.000_t1-t3.mat"),
     Drop_generator_6DMG(
         os.path.join(USE_CASE_RNN_COMPLETION_DIR,
                      "original_gram_files/gram_upperChar_all_sigma20.000_t1-t3.mat")))
]

np.random.seed(1)

for (direc, units, hidden_units, orig_gram_file_path, gen) in dataset_settings:
    k = 0
    for indices_to_drop in gen:
        k_dir = os.path.join(USE_CASE_RNN_COMPLETION_DIR,
                             direc,
                             str(units),
                             dataset_dir,
                             str(k))
        
        try:
            os.makedirs(k_dir)
        except FileExistsError:
            pass

        orig_gram_file = orig_gram_file_path.split("/")[-1]

        subprocess.run(["ln", "-s", orig_gram_file_path, k_dir])
        gram_file = os.path.join(k_dir, orig_gram_file)
        completionanalysisfile = gram_file.replace(".mat", ".error")

        json_dict = dict(gram_file=gram_file,
                         indices_to_drop=indices_to_drop,
                         completionanalysisfile=os.path.join(k_dir,
                                                             completionanalysisfile),
                         epochs=100,
                         patience=2,
                         units=units,
                         hidden_units=hidden_units)

        json_file_name = os.path.join(k_dir, "indices_to_drop.json")
        fd = open(json_file_name, "w")
        json.dump(json_dict, fd)
        fd.close()

        if os.uname().nodename.split('.')[0] in {'procyon', 'pollux', 'capella',
                                                 'aldebaran', 'rigel'}:
            job_file_name = os.path.join(k_dir, orig_gram_file + "_k" + str(k) + ".job")
            fd = open(job_file_name, "w")
            time_file_name = os.path.join(k_dir, "time_command.output")

            fd.write("echo $SHELL\n")
            fd.write("setenv LD_LIBRARY_PATH /home/ngym/NFSshare/tflib/lib64/:/home/ngym/NFSshare/tflib/usr/lib64/\n")
            fd.write("~/NFSshare/tflib/lib64/ld-2.17.so /usr/bin/time -v -o " +\
                     time_file_name + \
                     " ~/NFSshare/tflib/lib64/ld-2.17.so /usr/bin/python3 " +\
                     PROGRAM + " " + json_file_name + "\n")
            fd.close()
        k += 1
