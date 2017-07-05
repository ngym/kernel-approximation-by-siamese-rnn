import copy
import time

from fancyimpute import SoftImpute
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from pathos.multiprocessing import ProcessingPool

from algorithms import gak
from algorithms import siamese_rnn


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
    :returns: Filled in version of Gram matrix
    :rtype: list of lists, list of tuples
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
    start_time = time.time()
    not_indices = list(set(range(num_seqs)) - set(indices))
    for index in reversed(sorted(indices)):
        to_fill = [i for i in indices if i < index] + not_indices
        gram[index, to_fill] = pool.map(lambda j, i=index: gak.gak(seqs[i], seqs[j], sigma, triangular), to_fill)
        gram[index, index] = 1.
        gram[to_fill, index] = gram[index, to_fill].T
        num_finished_job += len(to_fill)
        current_time = time.time()
        duration_time = current_time - start_time
        eta = duration_time * num_job / num_finished_job - duration_time
        print("[%d/%d], %ds, ETA:%ds" % (num_finished_job, num_job, duration_time, eta), end='\r')
    end_time = time.time()
    print("[%d/%d], %ds, ETA:%ds" % (num_finished_job, num_job, duration_time, eta))
    pool.close()
    return gram, start_time, end_time


def softimpute_matrix_completion(gram_drop):
    """Fill in Gram matrix with dropped elements with Soft Impute Matrix Completion.
    Optimizes the Matrix Completion objective using Singular Value Thresholding

    :param gram_drop: Gram matrix with dropped elements
    :type gram_drop: list of lists
    :returns: Filled in Gram matrix, optimization start and end times
    :rtype: list of lists, float, float, float, float
    """
    t_start = time.time()
    gram_completed = SoftImpute().complete(gram_drop)
    t_end = time.time()
    return gram_completed, t_start, t_end


def rnn_matrix_completion(gram_drop, seqs,
                          epochs, patience,
                          logfile_loss, logfile_hdf5,
                          rnn,
                          rnn_units, dense_units,
                          dropout,
                          implementation,
                          bidirectional,
                          batchnormalization,
                          mode='train'):
    """Fill in Gram matrix with dropped elements with Keras Siamese RNN.
    Trains the network on given part of Gram matrix and the corresponding sequences
    Fills in missing elements by network prediction

    :param gram_drop: Gram matrix with dropped elements
    :param seqs: List of time series
    :param epochs: Number of passes over data set
    :param patience: Early Stopping parameter
    :param logfile_loss: Log file name for results
    :param logfile_hdf5: Log file name for network structure and weights in HDF5 format
    :param rnn_units: Recurrent layer sizes
    :param dense_units: Dense layer sizes
    :param rnn: Recurrent Layer type (Vanilla, LSTM or GRU)
    :param dropout: Dropout probability
    :param implementation: RNN implementation (0: CPU, 2: GPU, 1: any)
    :param bidirectional: Flag to switch between Forward and Bidirectional RNN
    :param batchnormalization: Flag to switch Batch Normalization on/off
    :param gram_drop: Keras Siamese RNN to be tested
    :param te_indices: Testing 2-tuples of time series index pairs
    :param gram_drop: Gram matrix with dropped elements
    :param seqs: List of time series
    :param load_pretrained: Flag to switch training from training set/use pretrained weights in HDF5 format
    :type gram_drop: np.ndarrays
    :type seqs: list of np.ndarrays
    :type epochs: int
    :type patience: int
    :type logfile_loss: str
    :type logfile_hdf5: str
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
    :rtype: list of lists, float, float, float, float
    """

    # pre-processing
    num_seqs = len(seqs)
    time_dim = max([seq.shape[0] for seq in seqs])
    pad_value = -123456789
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
                                  batchnormalization)

    # training
    # make 90% + 10% training validation random split
    trval_indices = np.random.permutation([(i, j)
                                           for i in range(num_seqs)
                                           for j in range(i, num_seqs)
                                           if not np.isnan(gram_drop[i][j])])
    tr_indices = trval_indices[:int(len(trval_indices) * 0.9)]
    val_indices = trval_indices[int(len(trval_indices) * 0.9):]
    tr_start = time.time()
    if mode == 'train':
        model.train_and_validate(tr_indices, val_indices,
                           gram_drop,
                           seqs,
                           epochs,
                           patience,
                           logfile_loss,
                           logfile_hdf5)
    elif mode == 'load_pretrained':
        print("load from hdf5 file: %s", logfile_hdf5)
        model.load_weights(logfile_hdf5)
    else:
        print('Unsupported mode.')
        exit -1
    tr_end = time.time()

    # prediction
    te_indices = [(i, j)
                  for i in range(num_seqs)
                  for j in range(i, num_seqs)
                  if np.isnan(gram_drop[i][j])]
    pred_start = time.time()
    preds = model.predict(te_indices, gram_drop, seqs)
    pred_end = time.time()

    # fill in
    gram_completed = gram_drop.tolist()
    for k in range(te_indices.__len__()):
        pred = preds[k][0]
        i, j = te_indices[k]
        assert np.isnan(gram_completed[i][j])
        gram_completed[i][j] = pred
        gram_completed[j][i] = pred
        assert not np.isnan(gram_completed[i][j])
    assert not np.any(np.isnan(np.array(gram_completed)))
    assert not np.any(np.isinf(np.array(gram_completed)))

    return gram_completed, tr_start, tr_end, pred_start, pred_end