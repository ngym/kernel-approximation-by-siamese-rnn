import sys, os, copy, time, json
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pickle
from collections import OrderedDict

import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, SimpleRNN, LSTM, GRU, Masking, Activation, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import Concatenate

from utils.nearest_positive_semidefinite import nearest_positive_semidefinite
from utils.errors import mean_squared_error, mean_absolute_error, relative_error
from utils.plot_gram_to_html import plot_gram_to_html
from utils.make_matrix_incomplete import gram_drop_random, gram_drop_samples
from datasets.read_sequences import read_sequences
from utils.multi_gpu import make_parallel


ngpus = 2

def create_RNN_base_network(input_shape, mask_value,
                            rnn_units=[5], dense_units=[2],
                            rnn="LSTM",
                            dropout=0.3,
                            implementation=2, bidirectional=False, batchnormalization=True):
    """Keras Deep RNN network to be used as Siamese branch.
    Stacks some Recurrent and some Dense layers on top of each other

    :param input_shape: Keras input shape
    :param mask_value: Padding value to be skipped among time steps
    :param rnn_units: Recurrent layer sizes
    :param dense_units: Dense layer sizes
    :param rnn: Recurrent Layer type (Vanilla, LSTM or GRU)
    :param dropout: Dropout probability
    :param implementation: RNN implementation (0: CPU, 2: GPU, 1: any)
    :param bidirectional: Flag to switch between Forward and Bidirectional RNN
    :param batchnormalization: Flag to switch Batch Normalization on/off
    :type input_shape: tuple
    :type mask_value: float
    :type rnn_units: list of ints
    :type dense_units: list of ints
    :type rnn: str
    :type dropout: float
    :type implementation: int
    :type bidirectional: bool
    :type batchnormalization: bool
    :returns: Keras Deep RNN network
    :rtype: keras.engine.training.Model
    """

    seq = Sequential()
    seq.add(Masking(mask_value=mask_value, input_shape=input_shape))

    if rnn == "Vanilla":
        r = SimpleRNN
    elif rnn == "LSTM":
        r = LSTM
    elif rnn == "GRU":
        r = GRU
    else:
        raise NotImplementedError("Currently rnn must be Vanilla, LSTM or GRU!")

    if bidirectional:
        b = Bidirectional
    else:
        b = lambda x: x

    for i in range(len(rnn_units)):
        rnn_unit = rnn_units[i]
        return_sequences = (i < (len(rnn_units) -1))
        seq.add(b(r(rnn_unit,
                    dropout=dropout, implementation=implementation,
                    return_sequences=return_sequences)))
        if batchnormalization and return_sequences:
            seq.add(BatchNormalization())
    for i in range(len(dense_units)):
        dense_unit = dense_units[i]
        seq.add(Dense(dense_unit, use_bias=False if batchnormalization else True))
        if batchnormalization:
            seq.add(BatchNormalization())
        seq.add(Activation('relu'))
    return seq

def generator_sequence_pairs(indices, gram_drop, seqs):
    """Siamese RNN data batch generator.
    Yields minibatches of 2 time series and their corresponding output value (Triangular Global Alignment kernel in our case)

    :param indices: 2-tuples of time series index pairs
    :param gram_drop: Gram matrix with dropped elements
    :param seqs: List of time series
    :type indices: list of tuples
    :type gram_drop: list of lists
    :type seqs: list of np.ndarrays
    :returns: Minibatch of data for Siamese RNN
    :rtype: list of np.ndarrays
    """

    indices_copy = copy.deepcopy(indices)
    batch_size = 512 * ngpus
    input_0 = []
    input_1 = []
    y = []
    for i, j in indices_copy:
        input_0.append(seqs[i])
        input_1.append(seqs[j])
        y.append([gram_drop[i][j]])
        if len(input_0) == batch_size:
            yield ([np.array(input_0), np.array(input_1)], np.array(y))
            input_0 = []
            input_1 = []
            y = []
    yield ([np.array(input_0), np.array(input_1)], np.array(y))

def train_and_validate(model, tr_indices, val_indices,
                       gram_drop,
                       seqs,
                       epochs,
                       patience,
                       logfile_loss,
                       logfile_hdf5):
    """Keras Siamese RNN training function.
    Carries out training and validation for given data over given number of epochs
    Logs results and network parameters

    :param model: Keras Siamese RNN to be trained
    :param tr_indices: Training 2-tuples of time series index pairs
    :param val_indices: Validation 2-tuples of time series index pairs
    :param gram_drop: Gram matrix with dropped elements
    :param seqs: List of time series
    :param epochs: Number of passes over data set
    :param patience: Early Stopping parameter
    :param logfile_loss: Log file name for results
    :param logfile_hdf5: Log file name for network structure and weights in HDF5 format
    :type model: keras.engine.training.Model
    :type tr_indices: list of tuples
    :type val_indices: list of tuples
    :type gram_drop: list of lists
    :type seqs: list of np.ndarrays
    :type epochs: int
    :type patience: int
    :type logfile_loss: str
    :type logfile_hdf5: str
    """
    
    fd_losses = open(logfile_loss, "w")
    
    list_ave_tr_loss = []
    list_tr_loss_batch = []
    list_ave_val_loss = []
    list_val_loss_batch = []
    wait = 0
    best_val_loss = np.inf
    fd_losses.write("epoch, num_batch_iteration, ave_tr_loss, tr_loss_batch, ave_val_loss, val_loss_batch\n")
    for epoch in range(1, epochs + 1):
        # training
        num_trained_samples = 0
        ave_tr_loss = 0
        np.random.shuffle(tr_indices)
        tr_gen = generator_sequence_pairs(tr_indices, gram_drop, seqs)
        tr_start = cur_time = time.time()
        num_batch_iteration = 0
        while num_trained_samples < len(tr_indices):
            # training batch
            x, y = next(tr_gen)
            tr_loss_batch = model.train_on_batch(x, y)
            ave_tr_loss = (ave_tr_loss * num_trained_samples + tr_loss_batch * y.shape[0]) / \
                       (num_trained_samples + y.shape[0])
            num_trained_samples += y.shape[0]
            prev_time = cur_time
            cur_time = time.time()
            print("epoch:[%d/%d] training:[%d/%d] %ds, ETA:%ds, ave_loss:%.5f, loss_batch:%.5f" %
                  (epoch, epochs, num_trained_samples,
                   len(tr_indices), cur_time - tr_start,
                   ((cur_time - prev_time) * len(tr_indices) / y.shape[0]) - (cur_time - tr_start),
                   ave_tr_loss, tr_loss_batch), end='\r')
            list_ave_tr_loss.append(ave_tr_loss)
            list_tr_loss_batch.append(tr_loss_batch)
            fd_losses.write("%d, %d, %.5f, %.5f, nan, nan\n" % (epoch, num_batch_iteration, ave_tr_loss, tr_loss_batch))
            fd_losses.flush()
            num_batch_iteration += 1
        print("epoch:[%d/%d] training:[%d/%d] %ds, ETA:%ds, ave_loss:%.5f, loss_batch:%.5f" %
              (epoch, epochs, num_trained_samples,
               len(tr_indices), cur_time - tr_start,
               ((cur_time - prev_time) * len(tr_indices) / y.shape[0]) - (cur_time - tr_start),
               ave_tr_loss, tr_loss_batch))

        # validation
        num_validated_samples = 0
        ave_val_loss = 0
        val_gen  = generator_sequence_pairs(val_indices, gram_drop, seqs)
        val_start = cur_time = time.time()
        while num_validated_samples < len(val_indices):
            # validation batch
            x, y = next(val_gen)
            val_loss_batch = model.test_on_batch(x,y)
            ave_val_loss = (ave_val_loss * num_validated_samples + val_loss_batch * y.shape[0]) / \
                       (num_validated_samples + y.shape[0])
            num_validated_samples += y.shape[0]
            prev_time = cur_time
            cur_time = time.time()
            print("                                                        epoch:[%d/%d] validation:[%d/%d] %ds, ETA:%ds, ave_loss:%.5f, loss_batch:%.5f" %
                  (epoch, epochs, num_validated_samples,
                   len(val_indices), cur_time - val_start,
                   ((cur_time - prev_time) * len(val_indices) / y.shape[0]) - (cur_time - val_start),
                   ave_val_loss, val_loss_batch), end='\r')
            list_ave_val_loss.append(ave_val_loss)
            list_val_loss_batch.append(val_loss_batch)
            fd_losses.write("%d, %d, nan, nan, %.5f, %.5f\n" % (epoch, num_batch_iteration, ave_val_loss, val_loss_batch))
            fd_losses.flush()
            num_batch_iteration += 1
        print("                                                        epoch:[%d/%d] validation:[%d/%d] %ds, ETA:%ds, ave_loss:%.5f, loss_batch:%.5f" %
              (epoch, epochs, num_validated_samples,
               len(val_indices), cur_time - val_start,
               ((cur_time - prev_time) * len(val_indices) / y.shape[0]) - (cur_time - val_start),
               ave_val_loss, val_loss_batch))
        if ave_val_loss < best_val_loss:
            best_val_loss = ave_val_loss
            model.save_weights(logfile_hdf5)
            best_weights = model.get_weights()
            wait = 0
        else:
            if wait >= patience:
                model.set_weights(best_weights)
                break
            wait += 1
    fd_losses.close()

def predict(model, te_indices, gram_drop, seqs):
    """Keras Siamese RNN prediction function.
    Carries out predicting for given data
    Logs results and network parameters

    :param model: Keras Siamese RNN to be used for prediction
    :param te_indices: Testing 2-tuples of time series index pairs
    :param gram_drop: Gram matrix with dropped elements
    :param seqs: List of time series
    :type model: keras.engine.training.Model
    :type te_indices: list of tuples
    :type gram_drop: list of lists
    :type seqs: list of np.ndarrays
    :returns: List of predicted network outputs
    :rtype: list of lists
    """
    
    te_gen = generator_sequence_pairs(te_indices, gram_drop, seqs)
    preds = []
    num_predicted_samples = 0
    while num_predicted_samples < len(te_indices):
        x, _ = next(te_gen)
        preds_batch = model.predict_on_batch(x)
        preds += preds_batch.tolist()
        num_predicted_samples += preds_batch.shape[0]
    return preds

def rnn_matrix_completion(gram_drop, seqs,
                          epochs, patience,
                          logfile_loss, logfile_hdf5,
                          rnn,
                          rnn_units, dense_units,
                          dropout,
                          implementation,
                          bidirectional,
                          batchnormalization):
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
    :type gram_drop: list of lists
    :type seqs: list of np.ndarrays
    :type epochs: int
    :type patience: int
    :type logfile_loss: str
    :type logfile_hdf5: str
    :type rnn: str
    :type rnn_units: int
    :type dense_units: int
    :type rnn: str
    :type dropout: float
    :type implementation: int
    :type bidirectional: bool
    :type batchnormalization: bool
    :returns: Filled in Gram matrix, training and prediction start and end times
    :rtype: list of lists, float, float, float, float
    """

    # pre-processing
    num_seqs = len(seqs)
    gram_drop = np.array(gram_drop)
    time_dim = max([seq.shape[0] for seq in seqs.values()])

    pad_value = -123456789
    seqs = pad_sequences([seq.tolist() for seq in seqs.values()],
                         maxlen=time_dim, dtype='float32',
                         padding='post', value=pad_value)

    feat_dim = seqs[0].shape[1]
    input_shape = (time_dim, feat_dim)

    # build network
    K.clear_session()
    base_network = create_RNN_base_network(input_shape, pad_value,
                                           rnn_units, dense_units,
                                           rnn,
                                           dropout,
                                           implementation,
                                           bidirectional,
                                           batchnormalization)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    con = Concatenate()([processed_a, processed_b])
    parent = Dense(units=1, use_bias=False if batchnormalization else True)(con)
    if batchnormalization:
        parent = BatchNormalization()(parent)
    out = Activation('sigmoid')(parent)

    model = Model([input_a, input_b], out)

    # training
    optimizer = Adam(clipnorm=1.)
    if ngpus > 1:
        model = make_parallel(model, ngpus)
    model.compile(loss='mse', optimizer=optimizer)
    # make 90% + 10% training validation random split
    trval_indices = np.random.permutation([(i, j)
                                        for i in range(num_seqs)
                                        for j in range(i, num_seqs)
                                        if not np.isnan(gram_drop[i][j])])
    tr_indices = trval_indices[:int(len(trval_indices) * 0.9)]
    val_indices = trval_indices[int(len(trval_indices) * 0.9):]
    tr_start = time.time()
    train_and_validate(model, tr_indices, val_indices,
                       gram_drop,
                       seqs,
                       epochs,
                       patience,
                       logfile_loss,
                       logfile_hdf5)
    tr_end = time.time()

    # prediction
    te_indices = [(i, j)
                  for i in range(num_seqs)
                  for j in range(i, num_seqs)
                  if np.isnan(gram_drop[i][j])]
    pred_start = time.time()
    preds = predict(model, te_indices, gram_drop, seqs)
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

def main():
    main_start = time.time()
    if len(sys.argv) != 2:
        random_drop = True
        gram_filename = sys.argv[1]
        if 'nipg' in os.uname().nodename:
            sample_dir = "~/shota/dataset"
        elif os.uname().nodename == 'atlasz' or 'cn' in os.uname().nodename:
            sample_dir = "/users/milacski/shota/dataset"
        elif os.uname().nodename == 'Regulus.local':
            sample_dir = "/Users/ngym/Lorincz-Lab/project/fast_time-series_data_classification/dataset"
        elif os.uname().nodename.split('.')[0] in {'procyon', 'pollux', 'capella',
                                                     'aldebaran', 'rigel'}:
            sample_dir = "/home/ngym/NFSshare/Lorincz_Lab/"
        else:
            sample_dir = sys.argv[4]
        drop_percent = int(sys.argv[2])
        completionanalysisfile = sys.argv[3]
        epochs = 2
        patience = 2
        rnn = "LSTM"
        rnn_units = [5]
        dense_units = [2]
        dropout = 0.3
        implementation = 2
        bidirectional = False
        batchnormalization = True
    else:
        random_drop = False
        config_json_file = sys.argv[1]
        config_dict = json.load(open(config_json_file, 'r'))

        gram_filename = config_dict['gram_file']
        sample_dir = config_dict['sample_dir']
        indices_to_drop = config_dict['indices_to_drop']
        completionanalysisfile = config_dict['completionanalysisfile']
        epochs = config_dict['epochs']
        patience = config_dict['patience']
        rnn = config_dict['rnn']
        rnn_units = config_dict['rnn_units']
        dense_units = config_dict['dense_units']
        dropout = config_dict['dropout']
        implementation = config_dict['implementation']
        bidirectional = config_dict['bidirectional']
        batchnormalization = config_dict['batchnormalization']

    fd = open(gram_filename, 'rb')
    pkl = pickle.load(fd)
    fd.close()
    
    dataset_type = pkl['dataset_type']
    gram_matrices = pkl['gram_matrices']
    if len(gram_matrices) == 1:
        gram = gram_matrices[0]['gram_original']
    else:
        gram = gram_matrices[-1]['gram_completed_npsd']
        
    #sample_names = [sn.replace(' ', '') for sn in pkl['sample_names']]
    sample_names = pkl['sample_names']

    logfile_loss = completionanalysisfile.replace(".timelog", ".losses")
    
    seqs = OrderedDict((k, v) for k, v in read_sequences(dataset_type, direc=sample_dir).items()
                       if k.split('/')[-1] in sample_names)
    
    seed = 1

    if random_drop:
        gram_drop, dropped_elements = gram_drop_random(seed, gram, drop_percent)
        logfile_html = gram_filename.replace(".pkl", "_drop" + str(drop_percent) + "_RNN_" + rnn + ".html")
        logfile_pkl  = gram_filename.replace(".pkl", "_drop" + str(drop_percent) + "_RNN_" + rnn + ".pkl")
        logfile_hdf5  = gram_filename.replace(".pkl", "_drop" + str(drop_percent) + "_RNN_" + rnn + ".hdf5")
    else:
        gram_drop, dropped_elements = gram_drop_samples(gram, indices_to_drop)
        logfile_html = gram_filename.replace(".pkl", "_dropfrom" + str(indices_to_drop[0]) + "_RNN_" + rnn + ".html")
        logfile_pkl  = gram_filename.replace(".pkl", "_dropfrom" + str(indices_to_drop[0]) + "_RNN_" + rnn + ".pkl")
        logfile_hdf5  = gram_filename.replace(".pkl", "_dropfrom" + str(indices_to_drop[0]) + "_RNN_" + rnn + ".hdf5")

    # RNN Completion
    gram_completed, fit_start, fit_end, pred_start, \
        pred_end = rnn_matrix_completion(gram_drop, seqs,
                                            epochs, patience,
                                            logfile_loss, logfile_hdf5,
                                            rnn,
                                            rnn_units, dense_units,
                                            dropout,
                                            implementation,
                                            bidirectional,
                                            batchnormalization)
    gram_completed = np.array(gram_completed)
    # eigenvalue check
    npsd_start = time.time()
    gram_completed_npsd = nearest_positive_semidefinite(gram_completed)
    npsd_end = time.time()

    # OUTPUT
    plot_gram_to_html(logfile_html,
                      gram_completed_npsd, sample_names)

    new_gram_matrices = {"gram_completed_npsd": np.array(gram_completed_npsd),
                         "gram_completed": np.array(gram_completed),
                         "gram_drop": np.array(gram_drop)}
    gram_matrices.append(new_gram_matrices)
    mat_log = pkl['log']
    new_log = "command: " + "".join(sys.argv) + time.asctime(time.localtime())
    mat_log.append(new_log)

    drop_indices = pkl['drop_indices']
    drop_indices.append(dropped_elements)

    pkl_fd = open(logfile_pkl, 'wb')
    dic = dict(gram_matrices=gram_matrices,
               drop_indices=drop_indices,
               dataset_type=dataset_type,
               log=mat_log,
               sample_names=sample_names)
    pickle.dump(dic, pkl_fd)
    pkl_fd.close()

    print("RNN Completion files are output.")

    mse = mean_squared_error(gram, gram_completed_npsd)
    msede = mean_squared_error(gram,
                               gram_completed_npsd,
                               dropped_elements)

    mae = mean_absolute_error(gram, gram_completed_npsd)
    maede = mean_absolute_error(gram,
                                gram_completed_npsd,
                                dropped_elements)
    
    re = relative_error(gram,
                        gram_completed_npsd)
    rede = relative_error(gram,
                         gram_completed_npsd,
                         dropped_elements)

    main_end = time.time()
    
    analysis_json = {}
    analysis_json['number_of_dropped_elements'] = len(dropped_elements)
    num_calculated_elements = len(dropped_elements) - len(indices_to_drop) // 2
    analysis_json['number_of_calculated_elements'] = num_calculated_elements
    analysis_json['fit_start'] = fit_start
    analysis_json['fit_end'] = fit_end
    analysis_json['fit_duration'] = fit_end - fit_start
    analysis_json['pred_start'] = pred_start
    analysis_json['pred_end'] = pred_end
    analysis_json['pred_duration'] = pred_end - pred_start
    analysis_json['npsd_start'] = npsd_start
    analysis_json['npsd_end'] = npsd_end
    analysis_json['npsd_duration'] = npsd_end - npsd_start
    analysis_json['main_start'] = main_start
    analysis_json['main_end'] = main_end
    analysis_json['main_duration'] = main_end - main_start
    analysis_json['mean_squared_error'] = mse
    analysis_json['mean_squared_error_of_dropped_elements'] = msede
    analysis_json['mean_absolute_error'] = mae
    analysis_json['mean_absolute_error_of_dropped_elements'] = maede
    analysis_json['relative_error'] = re
    analysis_json['relative_error_of_dropped_elements'] = rede

    fd = open(completionanalysisfile, "w")
    json.dump(analysis_json, fd)
    fd.close()

if __name__ == "__main__":
    main()

