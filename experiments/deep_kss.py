import os, sys, shutil

import numpy as np

from sacred import Experiment
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from datasets.read_sequences import read_sequences
from datasets.others import filter_samples

from datasets import k_fold_cross_validation
from algorithms import KSS_unsupervised_alpha_prediction 
from utils import file_utils
from utils import make_matrix_incomplete
from datasets.data_augmentation import augment_data, create_drop_flag_matrix

ex = Experiment('deep_kernel_group_lasso')

@ex.config
def cfg():
    lmbd = 1.
    output_dir = "results"
    fold_count = 5
    fold_to_test = 0
    fold_to_tv = [1,2]
    params = dict(epochs=100,
                  patience=2,
                  rnn="LSTM",
                  rnn_units=[10],
                  dense_units=[3],
                  dropout=0.3,
                  bidirectional=False,
                  batchnormalization=False,
                  implementation=1,
                  mode="train",
                  lmbd=lmbd,
                  top_activation="linear")
    # mode="load_pretrained"
    # mode="continue_training"

@ex.automain
def run(pickle_or_hdf5_location, dataset_location, fold_to_test, fold_to_tv,
        fold_count, params,
        output_dir, output_filename_format, data_augmentation_size):
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(os.path.abspath(sys.argv[2]), os.path.join(output_dir, os.path.basename(sys.argv[2])))
    hdf5 = pickle_or_hdf5_location[-4:] == "hdf5"
    if hdf5:
        loaded_data = file_utils.load_hdf5(os.path.abspath(pickle_or_hdf5_location))
    else:
        loaded_data = file_utils.load_pickle(os.path.abspath(pickle_or_hdf5_location))

    dataset_type = loaded_data['dataset_type']
    sample_names = [s.split('/')[-1].split('.')[0] for s in loaded_data['sample_names']]

    gram_matrices = loaded_data['gram_matrices']
    gram = gram_matrices[0]['original']
    
    sample_names = loaded_data['sample_names']
    
    folds = k_fold_cross_validation.get_kfolds(dataset_type, sample_names, fold_count)
    folds = np.array(folds)
    test_indices = np.concatenate(folds[fold_to_test])
    tv_indices = np.concatenate(folds[fold_to_tv])
    fold_for_gram = np.delete(np.arange(fold_count), fold_to_test + fold_to_tv)
    gram_indices = np.concatenate(folds[fold_for_gram]).astype(int)
    
    seqs, key_to_str, _ = read_sequences(dataset_type, direc=dataset_location)
    augmentation_magnification = 1.2
    seqs, key_to_str, flag_augmented = augment_data(seqs, key_to_str,
                                                    augmentation_magnification,
                                                    rand_uniform=True,
                                                    num_normaldist_ave=data_augmentation_size - 2)

    
    seqs = filter_samples(seqs, sample_names)
    key_to_str = filter_samples(key_to_str, sample_names)

    logfile_hdf5 = os.path.join(output_dir, output_filename_format + "_model.hdf5")
    logfile_loss = os.path.join(output_dir, output_filename_format + ".losses")
    output_file  = os.path.join(output_dir, output_filename_format + ".json")
    
    (roc_auc_score, f1_score) = KSS_unsupervised_alpha_prediction.get_classification_error(
        gram,
        gram_indices,
        tv_indices,
        test_indices,
        list(seqs.values()),
        params['epochs'],
        params['patience'],
        logfile_hdf5,
        logfile_loss,
        params['rnn'],
        params['rnn_units'],
        params['dense_units'],
        params['dropout'],
        params['implementation'],
        params['bidirectional'],
        params['batchnormalization'],
        params['mode'],
        list(key_to_str.values()),
        params['lmbd'],
        params['top_activation'])

    print(pickle_or_hdf5_location + " roc_auc_score: " + str(roc_auc_score) + " f1_score: " + str(f1_score))
    dic = dict(roc_auc_score=roc_auc_score,
               f1_score=f1_score)
    
    file_utils.save_json(output_file, dic)


    
