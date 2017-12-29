import sys, os, shutil
import time
from collections import OrderedDict

import numpy as np

from sacred import Experiment

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from algorithms import siamese_rnn_branch
from datasets.read_sequences import read_sequences
from datasets import k_fold_cross_validation
from utils import errors
from utils import file_utils
from utils import nearest_positive_semidefinite
from utils import make_matrix_incomplete
from algorithms import linear_svm

from datasets.read_sequences import read_sequences, pick_labels
from datasets.data_augmentation import augment_data, create_drop_flag_matrix
from datasets.others import filter_samples

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score, roc_auc_score

ex = Experiment('deep_feature_approximation_and_linear_svm')


@ex.config
def cfg():
    output_dir = "results/"
    params = None
    # The ratio of the size of the number of sequences
    # after data augmentation.
    # 1 means 1 times, hence no augmentation is applied.
    data_augmentation_size = 1
    # Split the data by fold_count
    # and treat fold_to_drop-th fold as test
    fold_count = 5
    fold_to_drop = 1
    params = dict(
        # Number of passes over data set
        epochs=100,
        # Early Stopping parameter
        patience=2,
        # Used in continued training
        epoch_start_from=1,
        # Recurrent Layer type (Vanilla, LSTM or GRU)
        rnn="LSTM",
        # Recurrent layer sizes
        rnn_units=[10,10],
        # Dense layer sizes
        dense_units=[3],
        # Dropout probability
        dropout=0.3,
        # Flag to switch between Forward and Bidirectional RNN
        bidirectional=False,
        # Flag to switch Batch Normalization on/off
        batchnormalization=False,
        # RNN implementation (0: CPU, 2: GPU, 1: any)
        implementation=1,
        # Mode to run (train, load_pretrained, continue_train)
        mode="train", 
        loss_function='mse',
        # The weight of in-domain kernel approximation in loss function
        loss_weight_ratio=10.0,
        # Implementation of the head of the Siamese network
        # (dense, dot_product, weighted_dot_product)
        siamese_joint_method="weighted_dot_product",
        # Activation of the top of each branch of the Siemese network
        siamese_arms_activation="linear",
        # Already trained model file
        trained_modelfile_hdf5=None)

@ex.capture
def check_fold(fold_count, fold_to_drop, hdf5):
    assert (0 <= fold_to_drop <= fold_count) or hdf5

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
def run(pickle_or_hdf5_location, dataset_location, fold_count, fold_to_drop,
        params, output_dir, output_filename_format,
        data_augmentation_size):
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(os.path.abspath(sys.argv[2]), os.path.join(output_dir, os.path.basename(sys.argv[2])))
    hdf5 = pickle_or_hdf5_location[-4:] == "hdf5"
    check_fold(fold_count, fold_to_drop, hdf5)

    pickle_or_hdf5_location = os.path.abspath(pickle_or_hdf5_location)
    dataset_location = os.path.abspath(dataset_location)
    output_dir = os.path.abspath(output_dir)
    assert os.path.isdir(output_dir)
    assert os.path.exists(pickle_or_hdf5_location)

    main_start = time.time()

    hdf5 = pickle_or_hdf5_location[-4:] == "hdf5"
    if hdf5:
        loaded_data = file_utils.load_hdf5(pickle_or_hdf5_location)
    else:
        loaded_data = file_utils.load_pickle(pickle_or_hdf5_location)

    dataset_type = loaded_data['dataset_type']
    if dataset_type == 'UCIauslan':
        sample_names = loaded_data['sample_names']
    else:
        sample_names = [s.split('/')[-1].split('.')[0] for s in loaded_data['sample_names']]
    gram_matrices = loaded_data['gram_matrices']
    if len(gram_matrices) == 1:
        gram = gram_matrices[0]['original']
    else:
        gram = gram_matrices[-1]['completed_npsd']

    # drop elements
    if fold_count == 0:        
        gram_drop = gram
    else:
        folds = k_fold_cross_validation.get_kfolds(dataset_type, sample_names, fold_count)
        indices_to_drop = folds[fold_to_drop - 1]
        gram_drop, dropped_elements = make_matrix_incomplete.gram_drop_samples(gram, indices_to_drop)

    seqs, key_to_str, _ = read_sequences(dataset_type, dataset_location)
    seqs = filter_samples(seqs, sample_names)
    key_to_str = filter_samples(key_to_str, sample_names)
    
    if data_augmentation_size > 1:
        augmentation_magnification = 1.2
        seqs, key_to_str, flag_augmented = augment_data(seqs, key_to_str,
                                                        augmentation_magnification,
                                                        rand_uniform=True,
                                                        num_normaldist_ave=data_augmentation_size - 2)

    train_start = None
    train_end = None

    modelfile_hdf5 = os.path.join(output_dir, output_filename_format + "_model.hdf5")
    logfile_loss = os.path.join(output_dir, output_filename_format + ".losses")
    model = siamese_rnn_branch.SiameseRNNBranch(gram_drop,
                                list(seqs.values()),
                                params['epochs'],
                                params['patience'],
                                params['epoch_start_from'],
                                logfile_loss,
                                modelfile_hdf5,
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
                                params['trained_modelfile_hdf5'],
                                params['siamese_arms_activation'])
    
    test_indices = indices_to_drop
    train_validation_indices = np.delete(np.arange(len(seqs)), test_indices)
    
    train_validation_seqs = seqs[train_validation_indices]
    test_seqs = seqs[test_indices]
    
    train_validation_features = model.predict(train_validation_seqs)
    test_features = model.predict(test_seqs)

    train_validation_labels = key_to_str[train_validation_indices]
    test_labels = key_to_str[test_indices]

    pred_end = time.time()
    
    auc, f1 = linear_svm.compute_classification_errors(train_validation_features,
                                                       train_validation_labels,
                                                       test_features,
                                                       test_labels)
    
    main_end = time.time()

    print("roc_auc:%f" % auc)
    print("f1:%f" % f1)





    
