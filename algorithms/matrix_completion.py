import os
import copy
import time

from fancyimpute import SoftImpute, KNN, IterativeSVD
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from pathos.multiprocessing import ProcessingPool

from algorithms import gak
from algorithms import siamese_rnn
from algorithms import siamese_rnn_branch

# TODO almost the same as in gak
def gak_matrix_completion(gram_drop, seqs, indices, sigma=None, triangular=None):
    """Fill in multiple rows and columns of Gram matrix.

    :param gram_drop: Gram matrix to be filled in
    :param seqs: List of time series to be used of filling in
    :param indices: Rows and columns to be filled in
    :param sigma: TGA kernel scale parameter
    :param triangular: TGA kernel band parameter
    :type gram_drop: np.ndarrays
    :type seqs: list of np.ndarrays
    :type indices: list of ints
    :type sigma: float
    :type triangular: int
    :returns: Filled in version of Gram matrix, complition start and end time
    :rtype: np.ndarray, float, float
    """

    gram = copy.deepcopy(gram_drop)

    if sigma is None:
        sigma = gak.calculate_gak_sigma(seqs)
    if triangular is None:
        triangular = gak.calculate_gak_triangular(seqs)

    pool = ProcessingPool()
    num_seqs = len(seqs)
    num_job = len(indices) * (num_seqs - len(indices)) + (len(indices) ** 2 - len(indices)) / 2
    num_finished_job = 0
    time_gak_start = os.times()
    not_indices = list(set(range(num_seqs)) - set(indices))
    for index in reversed(sorted(indices)):
        to_fill = [i for i in indices if i < index] + not_indices
        gram[index, to_fill] = pool.map(lambda j, i=index: gak.gak(seqs[i], seqs[j], sigma, triangular), to_fill)
        gram[index, index] = 1.
        gram[to_fill, index] = gram[index, to_fill].T
        num_finished_job += len(to_fill)
        time_current = os.times()
        duration_time = time_current[4] - time_gak_start[4]
        eta = duration_time * num_job / num_finished_job - duration_time
        print("[%d/%d], %ds, ETA:%ds" % (num_finished_job, num_job, duration_time, eta), end='\r')
    time_gak_end = os.times()
    print("[%d/%d], %ds, ETA:%ds" % (num_finished_job, num_job, duration_time, eta))
    pool.close()
    return gram, time_gak_start, time_gak_end


def softimpute_matrix_completion(gram_drop,
                                 seqs=None, sigma=None, triangular=None,
                                 num_process=4,
                                 drop_flag_matrix=None):
    """Fill in Gram matrix with dropped elements with Soft Impute Matrix Completion.
    Optimizes the Matrix Completion objective using Singular Value Thresholding

    :param gram_drop: Gram matrix with dropped elements
    :type gram_drop: list of lists
    :returns: Filled in Gram matrix, optimization start and end times
    :rtype: np.ndarray, float, float
    """
    time_completion_start = os.times()
    gram_completed = fancyimpute_matrix_completion("SoftImpute",
                                                   gram_drop,
                                                   seqs=seqs,
                                                   sigma=sigma,
                                                   triangular=triangular,
                                                   num_process=num_process,
                                                   drop_flag_matrix=drop_flag_matrix)
    time_completion_end = os.times()
    return gram_completed, time_completion_start, time_completion_end

def knn_matrix_completion(gram_drop,
                                 seqs=None, sigma=None, triangular=None,
                                 num_process=4,
                                 drop_flag_matrix=None):
    """Fill in Gram matrix with dropped elements with knn Matrix Completion.
    Optimizes the Matrix Completion objective using Singular Value Thresholding

    :param gram_drop: Gram matrix with dropped elements
    :type gram_drop: list of lists
    :returns: Filled in Gram matrix, optimization start and end times
    :rtype: np.ndarray, float, float
    """
    time_completion_start = os.times()
    #mean = np.mean(gram_drop, axis=1, keepdims=True)
    #std = np.std(gram_drop, axis=1, keepdims=True)
    #gram_drop_ = (gram_drop - mean) / (std + 1e-8)
    gram_drop_ = gram_drop
    gram_completed = fancyimpute_matrix_completion("KNN",
                                                   gram_drop_,
                                                   seqs=seqs,
                                                   sigma=sigma,
                                                   triangular=triangular,
                                                   num_process=num_process,
                                                   drop_flag_matrix=drop_flag_matrix)
    gram_completed_ = (gram_completed + gram_completed.T) / 2
    #gram_completed_ = (gram_completed * (std + 1e-8)) + mean
    time_completion_end = os.times()
    return gram_completed_, time_completion_start, time_completion_end

def iterativesvd_matrix_completion(gram_drop,
                                   seqs=None, sigma=None, triangular=None,
                                   num_process=4,
                                   drop_flag_matrix=None):
    """Fill in Gram matrix with dropped elements with IterativeSVD Matrix Completion.
    Optimizes the Matrix Completion objective using Singular Value Thresholding

    :param gram_drop: Gram matrix with dropped elements
    :type gram_drop: list of lists
    :returns: Filled in Gram matrix, optimization start and end times
    :rtype: np.ndarray, float, float
    """
    time_completion_start = os.times()
    mean = np.mean(gram_drop)
    std = np.std(gram_drop)
    gram_drop_ = (gram_drop - mean) /std
    gram_completed  = fancyimpute_matrix_completion("IterativeSVD",
                                                    gram_drop_,
                                                    seqs=seqs,
                                                    sigma=sigma,
                                                    triangular=triangular,
                                                    num_process=num_process,
                                                    drop_flag_matrix=drop_flag_matrix)
    gram_completed_ = (gram_completed * std) + mean
    time_completion_end = os.times()
    return gram_completed_, time_completion_start, time_completion_end

def fancyimpute_matrix_completion(function, gram_drop,
                                  seqs=None, sigma=None, triangular=None,
                                  num_process=4,
                                  drop_flag_matrix=None):
    gram_partially_completed_by_gak = gak.gram_gak(seqs,
                                                   sigma=sigma,
                                                   triangular=triangular,
                                                   num_process=num_process,
                                                   drop_flag_matrix=drop_flag_matrix)
    for i in range(len(gram_drop)):
        gram_drop[i, i] = 1
        for j in range(len(gram_drop[0])):
            if np.isnan(gram_partially_completed_by_gak[i, j]):
                continue
            assert np.isnan(gram_drop[i, j])
            gram_drop[i, j] = gram_partially_completed_by_gak[i, j]
    if function == "SoftImpute":
        gram_completed = SoftImpute().complete(gram_drop)
    elif function == "KNN":
        gram_completed = KNN().complete(gram_drop)
    elif function == "IterativeSVD":
        gram_completed = IterativeSVD().complete(gram_drop)
    else:
        print("unsupported fancyimpute functin")
        exit(-1)
    return gram_completed

def rnn_matrix_completion(gram_drop, seqs,
                          epochs, patience,
                          epoch_start_from,
                          logfile_loss,
                          new_modelfile_hdf5,
                          rnn,
                          rnn_units,
                          dense_units,
                          dropout,
                          implementation,
                          bidirectional,
                          batchnormalization,
                          mode,
                          loss_function,
                          loss_weight_ratio,
                          labels,
                          siamese_joint_method,
                          siamese_arms_activation,
                          trained_modelfile_hdf5=None):
    """Fill in Gram matrix with dropped elements with Keras Siamese RNN.
    Trains the network on given part of Gram matrix and the corresponding sequences
    Fills in missing elements by network prediction

    :param gram_drop: Gram matrix with dropped elements
    :param seqs: List of time series
    :param epochs: Number of passes over data set
    :param patience: Early Stopping parameter
    :param logfile_loss: Log file name for results
    :param modelfile_hdf5: Log file name for network structure and weights in HDF5 format
    :param rnn_units: Recurrent layer sizes
    :param dense_units: Dense layer sizes
    :param rnn: Recurrent Layer type (Vanilla, LSTM or GRU)
    :param dropout: Dropout probability
    :param implementation: RNN implementation (0: CPU, 2: GPU, 1: any)
    :param bidirectional: Flag to switch between Forward and Bidirectional RNN
    :param batchnormalization: Flag to switch Batch Normalization on/off
    :param gram_drop: Keras Siamese RNN to be tested
    :param test_indices: Testing 2-tuples of time series index pairs
    :param gram_drop: Gram matrix with dropped elements
    :param seqs: List of time series
    :param load_pretrained: Flag to switch training from training set/use pretrained weights in HDF5 format
    :type gram_drop: np.ndarrays
    :type seqs: list of np.ndarrays
    :type epochs: int
    :type patience: int
    :type logfile_loss: str
    :type modelfile_hdf5: str
    :type rnn: str
    :type rnn_units: list of int
    :type dense_units: list of int
    :type rnn: str
    :type dropout: float
    :type implementation: int
    :type bidirectional: bool
    :type batchnormalization: bool
    :type load_pretrained: bool
    :returns: Filled in Gram matrix, training and prediction start and end times
    :rtype: np.ndarray, float, float, float, float
    """

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
    model = siamese_rnn.SiameseRnn(input_shape, pad_value,
                                   rnn_units, dense_units,
                                   rnn,
                                   dropout,
                                   implementation,
                                   bidirectional,
                                   batchnormalization,
                                   loss_function,
                                   siamese_joint_method,
                                   siamese_arms_activation)

    # training
    # make 90% + 10% training validation random split
    train_and_validation_indices = np.random.permutation([(i, j)
                                            for i in range(num_seqs)
                                            for j in range(i, num_seqs)
                                            if not np.isnan(gram_drop[i][j])])
    train_indices = train_and_validation_indices[:int(len(train_and_validation_indices) * 0.9)]
    validation_indices = train_and_validation_indices[int(len(train_and_validation_indices) * 0.9):]
    time_train_start = os.times()
    if mode == 'train':
        assert epoch_start_from == 1
        model.train_and_validate(train_indices, validation_indices,
                                 gram_drop,
                                 seqs,
                                 labels,
                                 epochs,
                                 patience,
                                 epoch_start_from,
                                 loss_weight_ratio,
                                 logfile_loss,
                                 new_modelfile_hdf5)
    elif mode == 'continue_train':
        assert trained_modelfile_hdf5 != None
        print("load from hdf5 file: %s" % trained_modelfile_hdf5)
        model.load_weights(trained_modelfile_hdf5)
        model.train_and_validate(train_indices, validation_indices,
                                 gram_drop,
                                 seqs,
                                 labels,
                                 epochs,
                                 patience,
                                 epoch_start_from,
                                 loss_weight_ratio,
                                 logfile_loss,
                                 new_modelfile_hdf5)
    elif mode == 'fine_tuning':
        assert trained_modelfile_hdf5 != None
        print("Fine Tuning, load from hdf5 file: %s" % trained_modelfile_hdf5)
        model.load_weights(trained_modelfile_hdf5)
        model.train_and_validate(train_indices, validation_indices,
                                 gram_drop,
                                 seqs,
                                 labels,
                                 epochs,
                                 patience,
                                 epoch_start_from,
                                 loss_weight_ratio,
                                 logfile_loss,
                                 new_modelfile_hdf5)
    elif mode == 'load_pretrained':
        print("load from hdf5 file: %s" % trained_modelfile_hdf5)
        model.load_weights(trained_modelfile_hdf5)

    else:
        print('Unsupported mode.')
        exit(-1)
    time_train_end = os.times()

    # prediction
    test_indices = [(i, j)
                  for i in range(num_seqs)
                  for j in range(i, num_seqs)
                  if np.isnan(gram_drop[i][j])]
    time_pred_start = os.times()
    predictions = model.predict(test_indices, gram_drop, seqs)
    time_pred_end = os.times()

    # fill in
    gram_completed = copy.deepcopy(gram_drop)
    for k in range(len(test_indices)):
        prediction = predictions[k][0]
        i, j = test_indices[k]
        assert np.isnan(gram_completed[i][j])
        gram_completed[i][j] = prediction
        gram_completed[j][i] = prediction
        assert not np.isnan(gram_completed[i][j])
    assert not np.any(np.isnan(np.array(gram_completed)))
    assert not np.any(np.isinf(np.array(gram_completed)))

    return gram_completed, time_train_start, time_train_end,\
           time_pred_start, time_pred_end

def rapid_rnn_matrix_completion(gram_drop, seqs,
                                rnn,
                                rnn_units,
                                dense_units,
                                dropout,
                                implementation,
                                bidirectional,
                                batchnormalization,
                                loss_function,
                                siamese_arms_activation,
                                siamese_joint_method,
                                trained_modelfile_hdf5=None):
    """Fill in Gram matrix with dropped elements with Keras Siamese RNN.
    Trains the network on given part of Gram matrix and the corresponding sequences
    Fills in missing elements by network prediction

    :param gram_drop: Gram matrix with dropped elements
    :param seqs: List of time series
    :param epochs: Number of passes over data set
    :param patience: Early Stopping parameter
    :param logfile_loss: Log file name for results
    :param modelfile_hdf5: Log file name for network structure and weights in HDF5 format
    :param rnn_units: Recurrent layer sizes
    :param dense_units: Dense layer sizes
    :param rnn: Recurrent Layer type (Vanilla, LSTM or GRU)
    :param dropout: Dropout probability
    :param implementation: RNN implementation (0: CPU, 2: GPU, 1: any)
    :param bidirectional: Flag to switch between Forward and Bidirectional RNN
    :param batchnormalization: Flag to switch Batch Normalization on/off
    :param gram_drop: Keras Siamese RNN to be tested
    :param test_indices: Testing 2-tuples of time series index pairs
    :param gram_drop: Gram matrix with dropped elements
    :param seqs: List of time series
    :param load_pretrained: Flag to switch training from training set/use pretrained weights in HDF5 format
    :type gram_drop: np.ndarrays
    :type seqs: list of np.ndarrays
    :type epochs: int
    :type patience: int
    :type logfile_loss: str
    :type modelfile_hdf5: str
    :type rnn: str
    :type rnn_units: list of int
    :type dense_units: list of int
    :type rnn: str
    :type dropout: float
    :type implementation: int
    :type bidirectional: bool
    :type batchnormalization: bool
    :type load_pretrained: bool
    :returns: Filled in Gram matrix, training and prediction start and end times
    :rtype: np.ndarray, float, float, float, float
    """
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
                                                rnn_units,
                                                dense_units,
                                                rnn,
                                                dropout,
                                                implementation,
                                                bidirectional,
                                                batchnormalization,
                                                loss_function,
                                                siamese_joint_method,
                                                trained_modelfile_hdf5,
                                                siamese_arms_activation=siamese_arms_activation)
    
    test_indices = [i
                    for i in range(num_seqs)
                    if all(np.isnan(gram_drop[i]))]
    train_indices = np.delete(np.arange(len(seqs)), test_indices)
    
    train_seqs = seqs[train_indices]
    test_seqs = seqs[test_indices]
    
    train_features = model.predict(train_seqs)

    time_pred_start = os.times()
    test_features = model.predict(test_seqs)

    # fill in
    gram_completed = copy.deepcopy(gram_drop)
    for i, test_feature in zip(test_indices, test_features):
        for j, train_feature in zip(train_indices, train_features):
            prediction = np.inner(test_feature, train_feature)
            assert np.isnan(gram_completed[i][j])
            gram_completed[i][j] = prediction
            gram_completed[j][i] = prediction
            assert not np.isnan(gram_completed[i][j])
        for j, test_feature_ in zip(test_indices, test_features):
            if j > i:
                continue
            prediction = np.inner(test_feature, test_feature_)
            assert np.isnan(gram_completed[i][j])
            gram_completed[i][j] = prediction
            gram_completed[j][i] = prediction
            assert not np.isnan(gram_completed[i][j])
    time_pred_end = os.times()
    
    assert not np.any(np.isnan(np.array(gram_completed)))
    assert not np.any(np.isinf(np.array(gram_completed)))

    return gram_completed, time_pred_start, time_pred_end




