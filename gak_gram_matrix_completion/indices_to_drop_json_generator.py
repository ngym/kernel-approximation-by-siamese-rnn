import sys, json, os, subprocess

import numpy as np
import scipy as sp
from scipy import io

if os.uname().nodename == 'Regulus.local':
    USE_CASE_RNN_COMPLETION_DIR = "/Users/ngym/Lorincz-Lab/project/fast_time-series_data_classification/program/gak_gram_matrix_completion/USE_CASE_RNN_COMPLETION"
elif os.uname().nodename == 'nipgcore1':
    USE_CASE_RNN_COMPLETION_DIR = "/home/milacski/shota/USE_CASE_RNN_COMPLETION"
elif os.uname().nodename.split('.')[0] in {'procyon', 'pollux', 'capella', 'aldebaran', 'rigel'}:
    USE_CASE_RNN_COMPLETION_DIR = "/home/ngym/NFSshare/Lorincz_Lab/fast-time-series-data-classification/gak_gram_matrix_completion/USE_CASE_RNN_COMPLETION"
else:
    print("unsupported server")
    exit -1

original_gram_files = [
    ("UCIcharacter", os.path.join(USE_CASE_RNN_COMPLETION_DIR, "original_gram_files/gram_UCIcharacter_sigma20.000.mat")),
    ("UCItctodd", os.path.join(USE_CASE_RNN_COMPLETION_DIR, "original_gram_files/gram_UCItctodd_sigma12.000.mat")),
    ("6DMG", os.path.join(USE_CASE_RNN_COMPLETION_DIR, "original_gram_files/gram_upperChar_all_sigma20.000_t1-t3.mat"))]

np.random.seed(1)

for (direc, orig_gram_file_path) in original_gram_files:
    mat = io.loadmat(orig_gram_file_path)
    length = len(mat['gram'])

    bloch_size = length // 10

    permutated_indices = np.random.permutation([i for i in range(length)])

    for k in range(10):
        dataset_dir = os.path.join(USE_CASE_RNN_COMPLETION_DIR, direc)
        k_dir = os.path.join(dataset_dir, str(k))

        try:
            os.makedirs(k_dir)
        except FileExistsError:
            pass

        indices_to_drop = permutated_indices[k * bloch_size : (k+1) * bloch_size]
        orig_gram_file = orig_gram_file_path.split("/")[-1]

        subprocess.run(["ln", "-s", orig_gram_file_path, k_dir])
        gram_file = os.path.join(k_dir, orig_gram_file)
        completionanalysisfile = gram_file.replace(".mat", ".error")

        json_dict = dict(gram_file=gram_file,
                         indices_to_drop=indices_to_drop.tolist(),
                         completionanalysisfile=os.path.join(k_dir, completionanalysisfile),
                         epochs=100,
                         patience=2)

        json_file_name = os.path.join(k_dir, "indices_to_drop.json")
        fd = open(json_file_name, "w")
        json.dump(json_dict, fd)
        fd.close()

        if os.uname().nodename.split('.')[0] in {'procyon', 'pollux', 'capella', 'aldebaran', 'rigel'}:
            job_file_name = os.path.join(k_dir, orig_gram_file + "_k" + str(k) + ".job")
            fd = open(job_file_name, "w")
            time_file_name = os.path.join(k_dir, "time_command.output")

            fd.write("echo $SHELL\n")
            fd.write("setenv LD_LIBRARY_PATH /home/ngym/NFSshare/tflib/lib64/:/home/ngym/NFSshare/tflib/usr/lib64/\n")
            fd.write("~/NFSshare/tflib/lib64/ld-2.17.so /usr/bin/time -v -o " + time_file_name + \
                     "~/NFSshare/tflib/lib64/ld-2.17.so /usr/bin/python3 matrix_completion_rnn_residual.py " + job_file_name + "\n")
            fd.close()
