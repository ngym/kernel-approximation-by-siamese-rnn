import sys, os, shutil
import time
from collections import OrderedDict

import numpy as np

from sacred import Experiment

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from algorithms import matrix_completion
from datasets.read_sequences import read_sequences
from datasets import k_fold_cross_validation
from utils import errors
from utils import file_utils
from utils import nearest_positive_semidefinite
from utils import make_matrix_incomplete
from algorithms import KSS_unsupervised_alpha_prediction

from datasets.read_sequences import read_sequences, pick_labels
from datasets.data_augmentation import augment_data
from datasets.others import filter_samples

ex = Experiment('complete_matrix')


@ex.config
def cfg():
    output_dir = "results/"
    params = None
    fold_count = 5
    fold_to_drop = 1


@ex.named_config
def gak():
    algorithm = "gak"
    params = dict(sigma=20,
                  triangular=None)


@ex.named_config
def softimpute():
    algorithm = "softimpute"


@ex.named_config
def rnn():
    algorithm = "rnn"
    labels_to_use = []
    params = dict(epochs=100,
                  patience=2,
                  rnn="LSTM",
                  rnn_units=[10],
                  dense_units=[3],
                  dropout=0.3,
                  bidirectional=False,
                  batchnormalization=True,
                  implementation=1,
                  mode="train", # mode="load_pretrained"
                  loss_function='mse',
                  loss_weight_ratio=10.0,
                  siamese_joint_method="weighted_dot_product",
                  classify_one_by_all=False,
                  target_label="I")

@ex.capture
def check_algorithm(algorithm):
    assert algorithm in {"gak", "softimpute", "rnn"}


@ex.capture
def check_fold(fold_count, fold_to_drop, hdf5):
    assert (0 <= fold_to_drop <= fold_count) or hdf5


@ex.capture
def check_params(algorithm, params):
    if algorithm == "gak":
        if 'sigma' not in params:  # TODO else
            params['sigma'] = None
        if 'triangular' not in params:  # TODO else
            params['triangular'] = None

@ex.capture
def check_pickle_format(result_):
    result = list(result_.keys())
    assert "dataset_type" in result \
           and "gram_matrices" in result \
           and "drop_indices" in result \
           and "sample_names" in result \
           and "log" in result


def calculate_errors(gram, gram_completed_npsd, dropped_elements):
    mse = errors.mean_squared_error(gram, gram_completed_npsd)
    msede = errors.mean_squared_error(gram,
                                      gram_completed_npsd,
                                      dropped_elements)

    mae = errors.mean_absolute_error(gram, gram_completed_npsd)
    maede = errors.mean_absolute_error(gram,
                                       gram_completed_npsd,
                                       dropped_elements)

    re = errors.relative_error(gram,
                               gram_completed_npsd)
    rede = errors.relative_error(gram,
                                 gram_completed_npsd,
                                 dropped_elements)

    return mse, msede, mae, maede, re, rede


@ex.automain
def run(dataset_type, dataset_location, fold_count, fold_to_drop,
        algorithm, params, output_dir, output_filename_format,
        labels_to_use, data_augmentation_size):
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(os.path.abspath(sys.argv[2]), os.path.join(output_dir, os.path.basename(sys.argv[2])))
    check_algorithm(algorithm)
    check_params(algorithm, params)

    dataset_location = os.path.abspath(dataset_location)
    output_dir = os.path.abspath(output_dir)
    assert os.path.isdir(output_dir)

    main_start = time.time()

    seqs, key_to_str, _ = read_sequences(dataset_type, direc=dataset_location)
    sample_names = seqs.keys()
    
    gram = np.zeros([len(seqs), len(seqs)])

    # drop elements
    if fold_count == 0:
        gram_drop = gram
    else:
        folds = k_fold_cross_validation.get_kfolds(dataset_type, sample_names, fold_count)
        indices_to_drop = folds[fold_to_drop - 1]
        gram_drop, dropped_elements = make_matrix_incomplete.gram_drop_samples(gram, indices_to_drop)
    
    if data_augmentation_size > 1:
        if labels_to_use != []:
            seqs = pick_labels(dataset_type, seqs, labels_to_use)
        augmentation_magnification = 1.2
        seqs, key_to_str, flag_augmented = augment_data(seqs, key_to_str,
                                                        augmentation_magnification,
                                                        rand_uniform=True,
                                                        num_normaldist_ave=data_augmentation_size - 2)

    train_start = None
    train_end = None
    logfile_hdf5 = os.path.join(output_dir, output_filename_format + "_model.hdf5")
    logfile_loss = os.path.join(output_dir, output_filename_format + ".losses")
    gram_completed, train_start, train_end, completion_start, completion_end \
        = matrix_completion.rnn_matrix_completion(gram_drop,
                                                  list(seqs.values()),
                                                  params['epochs'],
                                                  params['patience'],
                                                  params['epoch_start_from'],
                                                  logfile_loss, logfile_hdf5,
                                                  params['rnn'],
                                                  params['rnn_units'],
                                                  params['dense_units'],
                                                  params['dropout'],
                                                  params['implementation'],
                                                  params['bidirectional'],
                                                  params['batchnormalization'],
                                                  params['mode'],
                                                  params['loss_function'],
                                                  params['loss_weight_ratio'],
                                                  list(key_to_str.values()),
                                                  params['siamese_joint_method'],
                                                  classify_one_by_all=params['classify_one_by_all'],
                                                  target_label=params['target_label'])
    action = "SiameseRNN"

    test_indices = indices_to_drop
    
    labels_list = list(key_to_str.values())
    labels = np.array(labels_list)
    pred_indices_in_gram = np.argmax(gram_completed[test_indices], axis=1)
    pred_labels = labels[pred_indices_in_gram]
    true_labels = labels[test_indices]

    labels_order = sorted(set(labels_list), key=labels_list.index)
    pred_indices = [labels_order.index(l) for l in pred_labels]
    true_indices = [labels_order.index(l) for l in true_labels]
    
    roc_auc_, f1_ = KSS_unsupervised_alpha_prediction.calc_scores(pred_indices, true_indices, len(labels_order))
    print("test roc_auc: %f" % roc_auc_)
    print("test f1     : %f" % f1_)










    
