import sys, os, shutil
import time
from collections import OrderedDict

import numpy as np

from sacred import Experiment

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from algorithms import rnn
from datasets.read_sequences import read_sequences
from datasets import k_fold_cross_validation
from utils import errors
from utils import file_utils
from utils import nearest_positive_semidefinite
from utils import make_matrix_incomplete
from algorithms import linear_svm

import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, SimpleRNN, LSTM, GRU, Masking, Activation, BatchNormalization
from keras.optimizers import RMSprop

from datasets.read_sequences import read_sequences, pick_labels
from datasets.data_augmentation import augment_data, create_drop_flag_matrix
from datasets.others import filter_samples

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score, roc_auc_score

ex = Experiment('End_to_End_RNN_and_SVM')


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

def main(dataset_type, dataset_location, fold_count, fold_to_drop,
        params, output_dir, output_filename_format, output_file,
        data_augmentation_size):
    os.makedirs(output_dir, exist_ok=True)
    try:
        shutil.copy(os.path.abspath(sys.argv[2]), os.path.join(output_dir, os.path.basename(sys.argv[2])))
    except shutil.SameFileError:
        pass

    dataset_location = os.path.abspath(dataset_location)
    output_dir = os.path.abspath(output_dir)
    assert os.path.isdir(output_dir)

    main_start = os.times()

    seqs, key_to_str, _ = read_sequences(dataset_type, dataset_location)

    print("%d samples." % len(seqs))
    
    if data_augmentation_size != 1:
        augmentation_magnification = 1.2
        seqs, key_to_str, flag_augmented = augment_data(seqs, key_to_str,
                                                        augmentation_magnification,
                                                        rand_uniform=True,
                                                        num_normaldist_ave=data_augmentation_size - 2)
    
    sample_names = list(seqs.keys())
    folds = k_fold_cross_validation.get_kfolds(dataset_type, sample_names, fold_count)
    indices_to_drop = folds[fold_to_drop - 1]

    modelfile_hdf5 = os.path.join(output_dir, output_filename_format + "_model.hdf5")

    # pre-processing
    seqs = seqs.values()
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
    rnn_ = rnn.Rnn(input_shape, pad_value,
                    params['rnn_units'],
                    params['dense_units'],
                    'tanh',
                    params['rnn'],
                    params['dropout'],
                    params['implementation'],
                    params['bidirectional'],
                    params['batchnormalization'])
    input_ = Input(shape=input_shape)
    base_network = rnn_.create_RNN_base_network()
    hidden = base_network(input_)
    output_ = Dense(len(set(list(key_to_str.values()))), activation='softmax')(hidden)
    model = Model(input_, output_)
    optimizer = RMSprop(clipnorm=1.)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    lb = LabelBinarizer()
    lb.fit(list(key_to_str.values()))
    Y = lb.transform(list(key_to_str.values()))

    test_indices = indices_to_drop
    train_validation_indices = np.delete(np.arange(len(seqs)), test_indices)
    
    train_validation_seqs = seqs[train_validation_indices]
    test_seqs = seqs[test_indices]

    Y_tr_val = Y[train_validation_indices]
    Y_test = Y[test_indices]

    callbacks = [
        ModelCheckpoint(modelfile_hdf5, save_best_only=True),
        EarlyStopping(patience=params['patience'])
    ]
    model.fit(train_validation_seqs, Y_tr_val, nb_epoch=params['epochs'], batch_size=512, verbose=1, callbacks=callbacks)

    time_pred_start = os.times()
    test_preds = model.predict(test_seqs)
    time_pred_end = os.times()

    main_end = os.times()

    roc_auc = roc_auc_score(y_true=Y_test, y_score=test_preds)
    f1 = f1_score(Y_test, test_preds, average='weighted')
    
    num_calculated_sequences = len(test_seqs)
    
    virtual_prediction_duration = time_pred_end.user - time_pred_start.user + time_pred_end.system - time_pred_start.system
    elapsed_prediction_duration = time_pred_end.elapsed - time_pred_start.elapsed

    virtual_classification_duration = 0
    elapsed_classification_duration = 0
    
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
    classification['basics']['roc_auc'] = roc_auc
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
    out_path = os.path.join(output_dir, output_file)
    file_utils.save_json(out_path, dic)

    
@ex.automain
def run(dataset_type, dataset_location, fold_count, fold_to_drop,
        params, output_dir, output_filename_format, output_file,
        data_augmentation_size):
    main(dataset_type, dataset_location, fold_count, fold_to_drop,
         params, output_dir, output_filename_format, output_file,
         data_augmentation_size)
    

    
