import sys, json, os, subprocess

import numpy as np
import scipy as sp
from scipy import io




""" Configuration
"""

if 'nipg' in os.uname().nodename:
    EXPERIMENTS_DIR = "~/shota/USE_CASE_RNN_COMPLETION"
elif os.uname().nodename == 'Regulus.local':
    EXPERIMENTS_DIR = "/Users/ngym/Lorincz-Lab/project/fast_time-series_data_classification/program/gak_gram_matrix_completion/USE_CASE_RNN_COMPLETION"
elif os.uname().nodename.split('.')[0] in {'procyon', 'pollux', 'capella', 'aldebaran', 'rigel'}:
    EXPERIMENTS_DIR = "/home/ngym/NFSshare/Lorincz_Lab/fast-time-series-data-classification/gak_gram_matrix_completion/USE_CASE_RNN_COMPLETION"
    PROGRAM = "/home/ngym/NFSshare/Lorincz_Lab/fast-time-series-data-classification/gak_gram_matrix_completion/matrix_completion_rnn_residual.py"
else:
    print("unsupported server")
    exit -1


""" Gram matrix .mat file paths
"""
gram_file_paths = {"UCIcharacter": os.path.join(EXPERIMENTS_DIR, "original_gram_files/gram_UCIcharacter_sigma20.000.mat"),
                   "UCIauslan": os.path.join(EXPERIMENTS_DIR, "original_gram_files/gram_UCIauslan_sigma12.000.mat"),
                   "gram_path": os.path.join(EXPERIMENTS_DIR, "original_gram_files/gram_upperChar_all_sigma20.000_t1-t3.mat")}

""" Configuration
"""
units_list = [(10, 3), (50, 15), (100, 33), (200, 66), (300, 100)]




"""Creates directories, symbolic links to mat files, k-fold cross-validation, json files, and timing for experiments.
"""
np.random.seed(1)
for dataset, mat_file_path in gram_file_paths.items():

    mat = io.loadmat(mat_file_path)
    length = len(mat['gram'])

    block_size = length // 10

    permutated_indices = np.random.permutation(length)

    for units, hidden_units in units_list:
        for k in range(10): # k-fold????
            dataset_dir = os.path.join(EXPERIMENTS_DIR, dataset, str(units))
            k_dir = os.path.join(dataset_dir, str(k))
            
            try:
                os.makedirs(k_dir)
            except FileExistsError:
                pass

            indices_to_drop = permutated_indices[k * block_size : (k+1) * block_size]
            mat_file_name = mat_file_path.split("/")[-1]

            subprocess.run(["ln", "-s", mat_file_path, k_dir])
            gram_file = os.path.join(k_dir, mat_file_name)
            completionanalysisfile = gram_file.replace(".mat", ".error")
            
            json_dict = dict(gram_file=gram_file,
                             indices_to_drop=indices_to_drop.tolist(),
                             completionanalysisfile=os.path.join(k_dir, completionanalysisfile),
                             epochs=100,
                             patience=2,
                             units=units,
                             hidden_units=hidden_units)
            
            json_file_name = os.path.join(k_dir, "indices_to_drop.json")
            fd = open(json_file_name, "w")
            json.dump(json_dict, fd)
            fd.close()

            if os.uname().nodename.split('.')[0] in {'procyon', 'pollux', 'capella', 'aldebaran', 'rigel'}:
                job_file_name = os.path.join(k_dir, mat_file_name + "_k" + str(k) + ".job")
                fd = open(job_file_name, "w")
                time_file_name = os.path.join(k_dir, "time_command.output")
                
                fd.write("echo $SHELL\n")
                fd.write("setenv LD_LIBRARY_PATH /home/ngym/NFSshare/tflib/lib64/:/home/ngym/NFSshare/tflib/usr/lib64/\n")
                fd.write("~/NFSshare/tflib/lib64/ld-2.17.so /usr/bin/time -v -o " + time_file_name + \
                         " ~/NFSshare/tflib/lib64/ld-2.17.so /usr/bin/python3 " + PROGRAM + " " + json_file_name + "\n")
                fd.close()
