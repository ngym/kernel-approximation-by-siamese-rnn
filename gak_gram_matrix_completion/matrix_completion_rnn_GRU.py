import sys, random
from collections import OrderedDict

import plotly.offline as po
import plotly.graph_objs as pgo

import numpy as np
import scipy as sp
from scipy import io
from scipy.io import wavfile
from scipy import signal

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import mean_squared_error

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, LSTM, Masking, Activation, BatchNormalization
from keras.optimizers import RMSprop
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping, History

from nearest_positive_semidefinite import nearest_positive_semidefinite
from mean_squared_error_of_dropped_elements import mean_squared_error_of_dropped_elements
from plot_gram_matrix import plot
from make_matrix_incomplete import make_matrix_incomplete
#import gc

import time, csv
from tempfile import mkdtemp
import gc
import os
import os.path as path

def batch_dot(vects):
    x, y = vects
    return K.batch_dot(x, y, axes=1)

def create_base_network(input_shape, mask_value):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(Masking(mask_value=mask_value, input_shape=input_shape))
    #seq.add(Dropout(0.1))
    #seq.add(LSTM(100, kernel_regularizer=l2(0.01), return_sequences=True))
    #seq.add(Dropout(0.1))
    seq.add(GRU(1000, kernel_regularizer=l2(0.01), dropout=0.1, implementation=2, return_sequences=False))
    seq.add(Dropout(0.1))
    seq.add(Dense(500, activation='linear', kernel_regularizer=l2(0.01)))
    seq.add(BatchNormalization())
    return seq

def generate_training_gak_pair(indices_list, incomplete_matrix, seqs):
    while True:
        for i, j in indices_list:
            if np.isnan(incomplete_matrix[i][j]):
                continue
            else:
                """
                print("i: %d, j:%d" % (i,j))
                print(seqs[i])
                print(seqs[j])
                print(incomplete_matrix[i][j])
                """
                yield ([np.array([seqs[i]]), np.array([seqs[j]])], [np.array([incomplete_matrix[i][j]])])
                # For training and validation, the next loop of while heppens
                # and the list gets shuffled.
                # For test data for prediction, shuffle does not get caused.
        np.random.shuffle(indices_list)

def generate_test_gak_pair(indices_list, incomplete_matrix, seqs):
    while True:
        for i, j in indices_list:
            if np.isnan(incomplete_matrix[i][j]):
                yield [np.array([seqs[i]]), np.array([seqs[j]])]
                # For training and validation, the next loop of while heppens
                # and the list gets shuffled.
                # For test data for prediction, shuffle does not get caused.


def rnn_matrix_completion(incomplete_matrix_, seqs_, files, fd, hdf5_out_rnn):
    incomplete_matrix = np.array(incomplete_matrix_)
    time_dim = max([seq_.shape[0] for seq_ in seqs_.values()])

    pad_value =  -123456789 # np.inf #np.nan # 0 #np.NINF #np.inf
    seqs = pad_sequences([s.tolist() for s in seqs_.values()],
                         maxlen=time_dim, dtype='float32',
                         padding='post', value=pad_value)

    feat_dim = seqs[0].shape[1]

    te_indices = []
    num_dropped = 0
    for i in range(len(files)):
        for j in range(i, len(files)):
            if np.isnan(incomplete_matrix[i][j]):
                te_indices.append((i, j))
                num_dropped += 1

    # network definition
    K.clear_session()
    
    input_shape = (time_dim, feat_dim)
    base_network = create_base_network(input_shape, pad_value)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    dot = Lambda(batch_dot)([processed_a, processed_b])
    out = Activation('sigmoid')(dot)

    model = Model([input_a, input_b], out)

    # train
    rms = RMSprop(clipnorm=1.)
    model.compile(loss='mse', optimizer=rms)
    model_checkpoint = ModelCheckpoint(hdf5_out_rnn, save_best_only=True)#, save_weights_only=True)
    early_stopping = EarlyStopping(patience=15)
    history = History()
    # need to pad train data nad validation data

    """
    generate random random permutation of (i,j)
    split 90 and 10
    make gentrain
    make genval
    """
    tv_indices = np.random.permutation([(i, j)
                                        for i in range(len(files))
                                        for j in range(i, len(files))
                                        if not np.isnan(incomplete_matrix[i][j])])
    tr_indices = tv_indices[:int(len(tv_indices) * 0.9)]
    v_indices = tv_indices[int(len(tv_indices) * 0.9):]
    tr_gen = generate_training_gak_pair(tr_indices, incomplete_matrix, seqs)
    v_gen = generate_training_gak_pair(v_indices, incomplete_matrix, seqs)
    
    fit_start = time.time()
    model.fit_generator(generator=tr_gen,
                        steps_per_epoch=len(tr_indices),
                        epochs=1, # 3 is enough for test, 300 would be proper for actual usage
                        validation_data=v_gen,
                        validation_steps=len(v_indices),
                        callbacks=[model_checkpoint, early_stopping, history])
                        
    fit_finish = time.time()
    fd.write("fit starts: " + str(fit_start))
    fd.write("\n")
    fd.write("fit finishes: " + str(fit_finish))
    fd.write("\n")
    fd.write("fit duration: " + str(fit_finish - fit_start))
    fd.write("\n")

    # need to pad test data
    # compute final result on test set
    #print(model.evaluate([te_pairs[:, 0, :, :], te_pairs[:, 1, :, :]], te_y))

    te_gen = generate_test_gak_pair(te_indices, incomplete_matrix, seqs)
    
    pred_start = time.time()
    preds = model.predict_generator(generator=te_gen,
                                    steps=len(te_indices))
    pred_finish = time.time()
    fd.write("pred starts: " + str(pred_start))
    fd.write("\n")
    fd.write("pred finishes: " + str(pred_finish))
    fd.write("\n")
    fd.write("pred duration: " + str(pred_finish - pred_start))
    fd.write("\n")
    print(preds)

    completed_matrix = incomplete_matrix.tolist()
    for k in range(te_indices.__len__()):
        pred = preds[k]
        i, j = te_indices[k]
        assert np.isnan(completed_matrix[i][j])
        completed_matrix[i][j] = pred
        completed_matrix[j][i] = pred
        assert not np.isnan(completed_matrix[i][j])
    assert not np.any(np.isnan(np.array(completed_matrix)))
    assert not np.any(np.isinf(np.array(completed_matrix)))

    return completed_matrix

def main():
    filename = sys.argv[1]
    incomplete_percentage = int(sys.argv[2])
    completionanalysisfile = sys.argv[3]
    mat = io.loadmat(filename)
    similarities = mat['gram']
    files = mat['indices']
    seqs = OrderedDict()

    fd = open(completionanalysisfile, "w")
    
    if filename.find("upperChar") != -1 or filename.find("velocity") != -1:
        for f in files:
            #print(f)
            if os.uname().nodename == 'atlasz':
                m = io.loadmat(f.replace("/home/ngym/NFSshare/Lorincz_Lab", "/users/milacski/shota/dataset"))
            elif os.uname().nodename == 'Regulus.local':
                m = io.loadmat(f.replace("/home/ngym/NFSshare/Lorincz_Lab", "/Users/ngym/Lorincz-Lab/project/fast_time-series_data_classification/dataset"))
            else:
                m = io.loadmat(f)
            seqs[f] = m['gest'].T
    elif filename.find("UCIcharacter") != -1:
        if os.uname().nodename == 'atlasz':
            datasetfile = "/users/milacski/shota/dataset/mixoutALL_shifted.mat"
        else:
            datasetfile = "/home/ngym/NFSshare/Lorincz_Lab/mixoutALL_shifted.mat"
        dataset = io.loadmat(datasetfile)
        displayname = [k[0] for k in dataset['consts']['key'][0][0][0]]
        classes = dataset['consts'][0][0][4][0]
        labels = []
        for c in classes:
            labels.append(displayname[c-1])
        i = 0
        for l in labels:
            seqs[l + str(i)] = dataset['mixout'][0][i].T
            i += 1
    elif filename.find("UCItctodd") != -1:
        for f in files:
            if os.uname().nodename == 'atlasz':
                reader = csv.reader(open(f.replace(' ', '')\
                                         .replace("/home/ngym/NFSshare/Lorincz_Lab", "/users/milacski/shota/dataset"),
                                     "r"), delimiter='\t')
            else:
                reader = csv.reader(open(f.replace(' ', ''), "r"), delimiter='\t')
            seq = []
            for r in reader:
                seq.append(r)
            seqs[f] = np.float64(np.array(seq))
    else:
        assert False

    seed = 1
        
    incomplete_similarities, dropped_elements = make_matrix_incomplete(seed, similarities, incomplete_percentage)

    fd.write("number of dropped elements: " + str(len(dropped_elements)))
    fd.write("\n")

    html_out_rnn = filename.replace(".mat", "_loss" + str(incomplete_percentage) + "_RNN.html")
    mat_out_rnn  = filename.replace(".mat", "_loss" + str(incomplete_percentage) + "_RNN.mat")
    hdf5_out_rnn  = filename.replace(".mat", "_hdf5" + str(incomplete_percentage) + "_RNN.hdf5")

    t_start = time.time()
    # "RnnCompletion"
    completed_similarities = np.array(rnn_matrix_completion(incomplete_similarities, seqs, files, fd, hdf5_out_rnn))
    # eigenvalue check
    psd_completed_similarities = nearest_positive_semidefinite(completed_similarities)
    t_finish = time.time()

    # OUTPUT
    plot(html_out_rnn,
         psd_completed_similarities, files)
    io.savemat(mat_out_rnn, dict(gram=psd_completed_similarities,
                                 dropped_gram=incomplete_similarities,
                                 indices=files))
    print("RnnCompletion files are output.")

    mse = mean_squared_error(similarities, psd_completed_similarities)
    msede = mean_squared_error_of_dropped_elements(similarities, psd_completed_similarities, dropped_elements)
    fd.write("start: " + str(t_start))
    fd.write("\n")
    fd.write("finish: " + str(t_finish))
    fd.write("\n")
    fd.write("duration: " + str(t_finish - t_start))
    fd.write("\n")
    fd.write("Mean squared error: " + str(mse))
    fd.write("\n")
    fd.write("Mean squared error of dropped elements: " + str(msede))
    fd.write("\n")
    fd.close()
    #gc.collect()

if __name__ == "__main__":
    main()
