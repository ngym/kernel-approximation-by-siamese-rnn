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

import keras.backend as K
from keras.preprocessing.sequence import pad_sequences

from datasets.read_sequences import read_sequences, pick_labels
from datasets.data_augmentation import augment_data
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
        params, output_dir, output_filename_format, output_file,
        data_augmentation_size):
    os.makedirs(output_dir, exist_ok=True)
    try:
        shutil.copy(os.path.abspath(sys.argv[2]), os.path.join(output_dir, os.path.basename(sys.argv[2])))
    except shutil.SameFileError:
        pass
    hdf5 = pickle_or_hdf5_location[-4:] == "hdf5"
    check_fold(fold_count, fold_to_drop, hdf5)

    pickle_or_hdf5_location = os.path.abspath(pickle_or_hdf5_location)
    dataset_location = os.path.abspath(dataset_location)
    output_dir = os.path.abspath(output_dir)
    assert os.path.isdir(output_dir)
    assert os.path.exists(pickle_or_hdf5_location)

    main_start = os.times()

    hdf5 = pickle_or_hdf5_location[-4:] == "hdf5"
    if hdf5:
        loaded_data = file_utils.load_hdf5(pickle_or_hdf5_location)
    else:
        loaded_data = file_utils.load_pickle(pickle_or_hdf5_location)

    dataset_type = loaded_data['dataset_type']
    if dataset_type == 'UCIauslan':
        loaded_sample_names = loaded_data['sample_names']
    else:
        loaded_sample_names = [s.split('/')[-1].split('.')[0] for s in loaded_data['sample_names']]
    gram_matrices = loaded_data['gram_matrices']
    if len(gram_matrices) == 1:
        gram = gram_matrices[0]['original']
    else:
        gram = gram_matrices[-1]['completed_npsd']

    # drop elements
    if fold_count == 0:        
        gram_drop = gram
    else:
        folds = k_fold_cross_validation.get_kfolds(dataset_type, loaded_sample_names, fold_count)
        indices_to_drop = folds[fold_to_drop - 1]
        gram_drop, dropped_elements = make_matrix_incomplete.gram_drop_samples(gram, indices_to_drop)

    seqs, sample_names, labels_str, _ = read_sequences(dataset_type, dataset_location)

    seqs = filter_samples(seqs, sample_names, loaded_sample_names)
    labels_str = filter_samples(labels_str, sample_names, loaded_sample_names)

    train_start = None
    train_end = None

    modelfile_hdf5 = os.path.join(output_dir, output_filename_format + "_model.hdf5")
    logfile_loss = os.path.join(output_dir, output_filename_format + ".losses")


    # pre-processing
    num_seqs = len(seqs)
    time_dim = max([seq.shape[0] for seq in seqs])
    pad_value = -4444
    seqs = pad_sequences([seq.tolist() for seq in seqs],
                         maxlen=time_dim, dtype='float32',
                         padding='post', value=pad_value)
    feat_dim = seqs[0].shape[1]
    input_shape = (time_dim, feat_dim)

    K.clear_session()

    # build network
    model = siamese_rnn_branch.SiameseRnnBranch(input_shape, pad_value,
                                                params['rnn_units'],
                                                params['dense_units'],
                                                params['rnn'],
                                                params['dropout'],
                                                params['implementation'],
                                                params['bidirectional'],
                                                params['batchnormalization'],
                                                params['loss_function'],
                                                params['siamese_joint_method'],
                                                params['trained_modelfile_hdf5'],
                                                siamese_arms_activation=params['siamese_arms_activation'])
    
    test_indices = indices_to_drop
    train_validation_indices = np.delete(np.arange(len(seqs)), test_indices)
    
    train_validation_seqs = seqs[train_validation_indices]
    test_seqs = seqs[test_indices]
    
    train_validation_features = model.predict(train_validation_seqs)

    time_pred_start = os.times()
    test_features = model.predict(test_seqs)
    time_pred_end = os.times()

    labels = np.array(labels_str)
    train_validation_labels = labels[train_validation_indices]
    test_labels = labels[test_indices]

    
    auc, f1, time_classification_start, time_classification_end = \
                    linear_svm.compute_classification_errors(train_validation_features,
                                                             train_validation_labels,
                                                             test_features,
                                                             test_labels)
    
    main_end = os.times()



    
    num_calculated_sequences = len(test_seqs)
    
    virtual_prediction_duration = time_pred_end.user - time_pred_start.user + time_pred_end.system - time_pred_start.system
    elapsed_prediction_duration = time_pred_end.elapsed - time_pred_start.elapsed

    virtual_classification_duration = time_classification_end.user - time_classification_start.user + time_classification_end.system - time_classification_start.system
    elapsed_classification_duration = time_classification_end.elapsed - time_classification_start.elapsed
    
    prediction = {}
    
    prediction['basics'] = {}
    prediction['basics']['number_of_calculated_sequences'] = len(test_seqs)
    
    prediction['all'] = {}
    prediction['all']['virtual_prediction_duration'] = virtual_prediction_duration
    prediction['all']['elapsed_prediction_duration'] = elapsed_prediction_duration
    
    prediction['each_seq'] = {}
    prediction['each_seq']['virtual_prediction_duration_per_calculated_sequence'] = virtual_prediction_duration / num_calculated_sequences
    prediction['each_seq']['elapsed_prediction_duration_per_calculated_sequence'] = elapsed_prediction_duration / num_calculated_sequences

    classification = {}

    classification['basics'] = {}
    classification['basics']['roc_auc'] = auc
    classification['basics']['f1'] = f1
    
    classification['all'] = {}
    classification['all']['virtual_classification_duration'] = virtual_classification_duration
    classification['all']['elapsed_classification_duration'] = elapsed_classification_duration
    
    classification['each_seq'] = {}
    classification['each_seq']['virtual_classification_duration_per_calculated_sequence'] = virtual_classification_duration / num_calculated_sequences
    classification['each_seq']['elapsed_classification_duration_per_calculated_sequence'] = elapsed_classification_duration / num_calculated_sequences
    
    dic = dict(prediction=prediction,
               classification=classification)
    
    ###
    lsvm_out_path = os.path.join(output_dir, output_file)
    file_utils.save_json(lsvm_out_path, dic)

    

    
