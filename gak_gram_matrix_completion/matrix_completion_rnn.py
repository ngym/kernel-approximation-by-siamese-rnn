import sys, random, copy, os, gc, time, csv
import os.path as path
from collections import OrderedDict
from tempfile import mkdtemp

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
from keras.layers import Dense, Dropout, Input, Lambda, LSTM, GRU, Masking, Activation, BatchNormalization
from keras.optimizers import RMSprop
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping, History

from nearest_positive_semidefinite import nearest_positive_semidefinite
from mean_squared_error_of_dropped_elements import mean_squared_error_of_dropped_elements
from plot_gram_matrix import plot
from make_matrix_incomplete import make_matrix_incomplete

from multi_gpu import make_parallel
ngpus = 2


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
    seq.add(LSTM(300, kernel_regularizer=l2(0.01), dropout=0.1, implementation=2, return_sequences=False))
    seq.add(Dropout(0.1))
    seq.add(Dense(100, activation='linear', kernel_regularizer=l2(0.01)))
    seq.add(BatchNormalization())
    return seq

def generator_sequence_pairs(indices_list_, incomplete_matrix, seqs):
    indices_list = copy.deepcopy(indices_list_)
    batch_size = 512 * ngpus
    input_0 = []
    input_1 = []
    y = []
    for i, j in indices_list:
        input_0.append(seqs[i])
        input_1.append(seqs[j])
        y.append([incomplete_matrix[i][j]])
        if len(input_0) == batch_size:
            yield ([np.array(input_0), np.array(input_1)], np.array(y))
            input_0 = []
            input_1 = []
            y = []
    yield ([np.array(input_0), np.array(input_1)], np.array(y))
    

def rnn_matrix_completion(incomplete_matrix_, seqs_, files, fd_analysis, fd_losses, hdf5_out_rnn):
    incomplete_matrix = np.array(incomplete_matrix_)
    time_dim = max([seq_.shape[0] for seq_ in seqs_.values()])

    pad_value =  -123456789 # np.inf #np.nan # 0 #np.NINF #np.inf
    seqs = pad_sequences([s.tolist() for s in seqs_.values()],
                         maxlen=time_dim, dtype='float32',
                         padding='post', value=pad_value)

    feat_dim = seqs[0].shape[1]

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
    
    if ngpus > 1:
        model = make_parallel(model,ngpus)
    model.compile(loss='mse', optimizer=rms)
    #model_checkpoint = ModelCheckpoint(hdf5_out_rnn, save_best_only=True)#, save_weights_only=True)
    #early_stopping = EarlyStopping(patience=15)
    #history = History()
    # need to pad train data nad validation data

    """
    generate random random permutation of (i,j)
    split 90 and 10
    make gentrain
    make genval
    """
    # for choose validation data at random
    tv_indices = np.random.permutation([(i, j)
                                        for i in range(len(files))
                                        for j in range(i, len(files))
                                        if not np.isnan(incomplete_matrix[i][j])])
    tr_indices = tv_indices[:int(len(tv_indices) * 0.9)]
    v_indices = tv_indices[int(len(tv_indices) * 0.9):]

    patience = 10
    epochs = 2

    list_ave_tr_loss = []
    list_tr_loss_batch = []
    list_ave_v_loss = []
    list_v_loss_batch = []
    fit_start = time.time()
    wait = 0
    best_sum_validation_loss = np.inf
    for epoch in range(1, epochs + 1):
        num_trained_samples = 0
        ave_tr_loss = 0
        np.random.shuffle(tr_indices)
        tr_gen = generator_sequence_pairs(tr_indices, incomplete_matrix, seqs)
        cur_time = time.time()
        num_iterated = 0
        while num_trained_samples < len(tr_indices):
            # training
            x, y = next(tr_gen)
            tr_loss_batch = model.train_on_batch(x, y)
            ave_tr_loss = (ave_tr_loss * num_trained_samples + tr_loss_batch * y.shape[0]) / \
                       (num_trained_samples + y.shape[0])
            num_trained_samples += y.shape[0]
            prev_time = cur_time
            cur_time = time.time()
            print("epoch:[%d/%d] training:[%d/%d] %ds, ETA:%ds, ave_loss:%.5f, loss_batch:%.5f" %
                  (epoch, epochs, num_trained_samples,
                   len(tr_indices), cur_time - fit_start,
                   ((cur_time - prev_time) * len(tr_indices) / y.shape[0]) - (cur_time - fit_start),
                   ave_tr_loss, tr_loss_batch), end='\r')
            list_ave_tr_loss.append(ave_tr_loss)
            list_tr_loss_batch.append(tr_loss_batch)
            fd_losses.write("%d,%d,%.5f,%.5f,%5f,%5f" % (epoch, num_iterated, ave_tr_loss, tr_loss_batch, 0, 0))
            num_iterated += 1
        print("epoch:[%d/%d] training:[%d/%d] %ds, ETA:%ds, ave_loss:%.5f, loss:batch:%.5f" %
              (epoch, epochs, num_trained_samples,
               len(tr_indices), cur_time - fit_start,
               ((cur_time - prev_time) * len(tr_indices) / y.shape[0]) - (cur_time - fit_start),
               ave_tr_loss, tr_loss_batch), end='\r')

        num_validated_samples = 0
        ave_v_loss = 0
        best_v_loss = np.inf
        v_gen  = generator_sequence_pairs(v_indices, incomplete_matrix, seqs)
        cur_time = time.time()
        while num_validated_samples < len(v_indices):
            # validation
            x, y = next(v_gen)
            v_loss_batch = model.test_on_batch(x,y)
            ave_v_loss = (ave_v_loss * num_validated_samples + v_loss_batch * y.shape[0]) / \
                       (num_validated_samples + y.shape[0])
            num_validated_samples += y.shape[0]
            prev_time = cur_time
            cur_time = time.time()
            print("                                                        epoch:[%d/%d] validation:[%d/%d] %ds, ETA:%ds, ave_loss:%.5f, loss_batch:%.5f" %
                  (epoch, epochs, num_validated_samples,
                   len(v_indices), cur_time - fit_start,
                   ((cur_time - prev_time) * len(v_indices) / y.shape[0]) - (cur_time - fit_start),
                   ave_v_loss, v_loss_batch), end='\r')
            list_ave_v_loss.append(ave_v_loss)
            list_v_loss_batch.append(v_loss_batch)
            fd_losses.write("%d,%d,%.5f,%.5f,%5f,%5f" % (epoch, num_iterated, 0, 0, ave_v_loss, v_loss_batch))
            num_iterated += 1
        print("                                                        epoch:[%d/%d] validation:[%d/%d] %ds, ETA:%ds, ave_loss:%.5f, loss_batch:%.5f" %
              (epoch, epochs, num_validated_samples,
               len(v_indices), cur_time - fit_start,
               ((cur_time - prev_time) * len(v_indices) / y.shape[0]) - (cur_time - fit_start),
               ave_v_loss, v_loss_batch), end='\r')
        if ave_v_loss < best_v_loss:
            best_v_loss = ave_v_loss
            model.save_weights(hdf5_out_rnn)
            best_weights = model.get_weights()
            wait = 0
        else:
            if wait >= patience:
                model.set_weights(best_weights)
                break
            wait += 1
                        
    fit_finish = time.time()
    fd_analysis.write("fit starts: " + str(fit_start))
    fd_analysis.write("\n")
    fd_analysis.write("fit finishes: " + str(fit_finish))
    fd_analysis.write("\n")
    fd_analysis.write("fit duration: " + str(fit_finish - fit_start))
    fd_analysis.write("\n")

    # need to pad test data
    # compute final result on test set
    #print(model.evaluate([te_pairs[:, 0, :, :], te_pairs[:, 1, :, :]], te_y))

    te_indices = [(i, j)
                  for i in range(len(files))
                  for j in range(i, len(files))
                  if np.isnan(incomplete_matrix[i][j])]
    #num_dropped = len(te_indices)
    
    te_gen = generator_sequence_pairs(te_indices, incomplete_matrix, seqs)
    
    pred_start = time.time()
    preds = []
    num_predicted_samples = 0
    while num_predicted_samples < len(te_indices):
        x, _ = next(te_gen)
        preds_batch = model.predict_on_batch(x)
        preds += preds_batch.tolist()
        num_predicted_samples += preds_batch.shape[0]
    
    pred_finish = time.time()
    fd_analysis.write("pred starts: " + str(pred_start))
    fd_analysis.write("\n")
    fd_analysis.write("pred finishes: " + str(pred_finish))
    fd_analysis.write("\n")
    fd_analysis.write("pred duration: " + str(pred_finish - pred_start))
    fd_analysis.write("\n")

    completed_matrix = incomplete_matrix.tolist()
    for k in range(te_indices.__len__()):
        pred = preds[k][0]
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

    fd_analysis = open(completionanalysisfile, "w")
    lossesfile = completionanalysisfile.replace(".error", "losses")
    fd_losses = open(lossesfile, "w")
    
    if filename.find("upperChar") != -1 or filename.find("velocity") != -1:
        for f in files:
            #print(f)
            if os.uname().nodename == 'atlasz' or 'cn' in os.uname().nodename:
                m = io.loadmat(f.replace("/home/ngym/NFSshare/Lorincz_Lab",
                                         "/users/milacski/shota/dataset"))
            elif os.uname().nodename == 'nipgcore1':
                m = io.loadmat(f.replace("/home/ngym/NFSshare/Lorincz_Lab",
                                         "/home/milacski/shota/dataset"))
            elif os.uname().nodename == 'Regulus.local':
                m = io.loadmat(f.replace("/home/ngym/NFSshare/Lorincz_Lab",
                                         "/Users/ngym/Lorincz-Lab/project/fast_time-series_data_classification/dataset"))
            else:
                m = io.loadmat(f)
            seqs[f] = m['gest'].T
    elif filename.find("UCIcharacter") != -1:
        if os.uname().nodename == 'atlasz' or 'cn' in os.uname().nodename:
            datasetfile = "/users/milacski/shota/dataset/mixoutALL_shifted.mat"
        elif os.uname().nodename == 'nipgcore1':
            datasetfile = "/home/milacski/shota/dataset/mixoutALL_shifted.mat"
        elif os.uname().nodename == 'Regulus.local':
            datasetfile = "/Users/ngym/Lorincz-Lab/project/fast_time-series_data_classification/dataset/UCI/mixoutALL_shifted.mat"
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
            if os.uname().nodename == 'atlasz' or 'cn' in os.uname().nodename:
                reader = csv.reader(open(f.replace(' ', '')\
                                         .replace("/home/ngym/NFSshare/Lorincz_Lab",
                                                  "/users/milacski/shota/dataset"),
                                     "r"), delimiter='\t')
            elif os.uname().nodename == 'nipgcore1':
                reader = csv.reader(open(f.replace(' ', '')\
                                         .replace("/home/ngym/NFSshare/Lorincz_Lab",
                                                  "/home/milacski/shota/dataset"),
                                    "r"), delimiter='\t')
            elif os.uname().nodename == 'Regulus.local':
                reader = csv.reader(open(f.replace(' ', '')\
                                         .replace("/home/ngym/NFSshare/Lorincz_Lab",
                                                  "/Users/ngym/Lorincz-Lab/project/fast_time-series_data_classification/dataset"),
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

    fd_analysis.write("number of dropped elements: " + str(len(dropped_elements)))
    fd_analysis.write("\n")

    html_out_rnn = filename.replace(".mat", "_loss" + str(incomplete_percentage) + "_RNN.html")
    mat_out_rnn  = filename.replace(".mat", "_loss" + str(incomplete_percentage) + "_RNN.mat")
    hdf5_out_rnn  = filename.replace(".mat", "_hdf5" + str(incomplete_percentage) + "_RNN.hdf5")

    t_start = time.time()
    # "RnnCompletion"
    completed_similarities = np.array(rnn_matrix_completion(incomplete_similarities, seqs, files, fd_analysis, fd_losses, hdf5_out_rnn))
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
    fd_analysis.write("start: " + str(t_start))
    fd_analysis.write("\n")
    fd_analysis.write("finish: " + str(t_finish))
    fd_analysis.write("\n")
    fd_analysis.write("duration: " + str(t_finish - t_start))
    fd_analysis.write("\n")
    fd_analysis.write("Mean squared error: " + str(mse))
    fd_analysis.write("\n")
    fd_analysis.write("Mean squared error of dropped elements: " + str(msede))
    fd_analysis.write("\n")
    fd_analysis.close()
    #gc.collect()

if __name__ == "__main__":
    main()
