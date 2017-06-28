import sys, json, os, subprocess, pickle
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datasets.k_fold_cross_validation_generators import KFold_UCIauslan, KFold, KFold_6DMGupperChar

""" Configuration
"""

if 'nipg' in os.uname().nodename:
    EXPERIMENTS_DIR = "/home/milacski/shota/USE_CASE_RNN_COMPLETION_1_VALIDATION"
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
        kfold = KFold_6DMGupperChar
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

            ### training to construct network and test
            json_file_name = os.path.join(k_dir, "config_rnn_completion.json")
            fd = open(json_file_name, "w")
            json.dump(json_dict, fd)
            fd.close()

            ### use pretrained hdf5 file for constructing network and test
            json_file_name = os.path.join(k_dir, "config_rnn_completion_pretraining.json")
            json_dict['pretraining'] = True
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
                time_file_name = os.path.join(k_dir, "time_command_pretraining.output")
                fd = open(command_file_name, "w")
                fd.write(TIME + " -v -o " + time_file_name +\
                         " python3 " +\
                         PROGRAM + " " + json_file_name + "\n")
                fd.close()

            k += 1

