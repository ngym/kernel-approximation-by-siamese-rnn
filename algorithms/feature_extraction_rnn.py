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

from matrix_completion_rnn import create_RNN_siamese_network

ngpus = 2

def generator_sequences(seqs):
    batch_size = 512 * ngpus
    input = []
    for i in range(len(seqs)):
        input.append(seqs[i])
        if len(input) == batch_size:
            yield np.array(input)
            input = []
    yield np.array(input)

def predict(model, seqs):
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
    gen = generator_sequences(seqs)
    preds = []
    num_predicted_samples = 0
    while num_predicted_samples < len(seqs):
        x = next(gen)
        preds_batch = model.predict_on_batch(x)
        preds += preds_batch.tolist()
        num_predicted_samples += preds_batch.shape[0]
    return preds
    
def rnn_feature_extraction(seqs,
                           logfile_hdf5,
                           rnn,
                           rnn_units, dense_units,
                           dropout,
                           implementation,
                           bidirectional,
                           batchnormalization):
    # pre-processing
    num_seqs = len(seqs)
    time_dim = max([seq.shape[0] for seq in seqs.values()])
    pad_value = -123456789
    seqs = pad_sequences([seq.tolist() for seq in seqs.values()],
                         maxlen=time_dim, dtype='float32',
                         padding='post', value=pad_value)
    feat_dim = seqs[0].shape[1]
    input_shape = (time_dim, feat_dim)
    
    K.clear_session()
    
    # build network    
    print("load from hdf5 file: %s", logfile_hdf5)
    model = create_RNN_siamese_network(input_shape, pad_value,
                                       rnn_units, dense_units,
                                       rnn,
                                       dropout,
                                       implementation,
                                       bidirectional,
                                       batchnormalization)
    model.load_weights(logfile_hdf5)
    new_model = Model(model.get_layer('base_input'),
                      model.get_layer('model_1').get_layer('sequential_1').get_layer('base_hidden').output)
    model = new_model

    pred_start = time.time()
    preds = predict(model, seqs)
    pred_end = time.time()
    return preds, pred_start, pred_end

def main():
    main_start = time.time()
    mode = 'feature_extraction'
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
            sample_dir = "/Users/ngym/Lorincz-Lab/project/fast_time-series_data_classification/USE_CASE_RNN_MATRIX_COMPLETION/original_gram_files/UCIcharacter"
        else:
            sample_dir = sys.argv[3]
        indices_to_drop = 0
        completionanalysisfile = sys.argv[2]
        epochs = 2
        patience = 2
        rnn = "LSTM"
        rnn_units = [10]
        dense_units = [3]
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
        if 'mode' in config_dict.keys():
            mode = config_dict['mode']

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

    logfile_pkl  = gram_filename.replace(".pkl", "_dropfrom" + str(indices_to_drop[0]) + "_RNN_" + rnn + ".pkl")
    logfile_hdf5  = gram_filename.replace(".pkl", "_dropfrom" + str(indices_to_drop[0]) + "_RNN_" + rnn + ".hdf5")

    # RNN Completion
    preds, pred_start, pred_end = rnn_feature_extraction(seqs,
                                                         logfile_hdf5,
                                                         rnn,
                                                         rnn_units, dense_units,
                                                         dropout,
                                                         implementation,
                                                         bidirectional,
                                                         batchnormalization)

    mat_log = pkl['log']
    new_log = "command: " + "".join(sys.argv) + time.asctime(time.localtime())
    mat_log.append(new_log)
    
    pkl_fd = open(logfile_pkl, 'wb')
    dic = dict(extracted_features=preds,
               dataset_type=dataset_type,
               log=mat_log,
               sample_names=sample_names)
    pickle.dump(dic, pkl_fd)
    pkl_fd.close()

    print("RNN feature extraction files are output.")

    main_end = time.time()
    
    analysis_json = {}
    analysis_json['number_of_samples'] = len(seqs)
    analysis_json['pred_start'] = pred_start
    analysis_json['pred_end'] = pred_end
    analysis_json['pred_duration'] = pred_end - pred_start
    analysis_json['main_start'] = main_start
    analysis_json['main_end'] = main_end
    analysis_json['main_duration'] = main_end - main_start

    fd = open(completionanalysisfile, "w")
    json.dump(analysis_json, fd)
    fd.close()

if __name__ == "__main__":
    main()


