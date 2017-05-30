import sys, random

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
from keras.layers import Dense, Dropout, Input, Lambda, LSTM, Masking
from keras.optimizers import RMSprop
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping, History

    
def mean_squared_error_of_dropped_elements(m1, m2, elements):
    m1 = np.array(m1)
    m2 = np.array(m2)
    assert m1.shape == m2.shape
    sum_squared_error = 0
    for i, j in elements:
        sum_squared_error += (m1[i][j] - m2[i][j]) ** 2
    return sum_squared_error / elements.__len__()

def batch_dot(vects):
    x, y = vects
    return K.batch_dot(x, y, axes=1)

def create_base_network(input_shape, mask_value):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(Masking(mask_value=mask_value, input_shape=input_shape))
    seq.add(Dropout(0.1))
    #seq.add(LSTM(128, return_sequences=True))
    #seq.add(Dropout(0.1))
    seq.add(LSTM(128, return_sequences=False))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='linear', kernel_regularizer=l2(0.01)))
    return seq

def rnn_matrix_completion(incomplete_matrix_, seqs_, files):
    incomplete_matrix = np.array(incomplete_matrix_)
    time_dim = max([seq_.shape[0] for seq_ in seqs_.values()])

    pad_value =  -123456789 # np.inf #np.nan # 0 #np.NINF #np.inf
    seqs = pad_sequences([s.tolist() for s in seqs_.values()],
                         maxlen=time_dim, dtype='float32',
                         padding='post', value=pad_value)

    feat_dim = seqs[0].shape[1]

    tr_pairs = []
    tr_y = []
    te_pairs = []
    te_pairs_index = []
    for i in range(files.__len__()):
        for j in range(files.__len__()):
            if np.isnan(incomplete_matrix[i][j]):
                # test
                te_pairs += [[seqs[i], seqs[j]]]
                te_pairs_index += [[i, j]]
    while len(tr_pairs) < 50000:
        i = random.randint(0, len(files)-1)
        j = random.randint(0, len(files)-1)
        if not np.isnan(incomplete_matrix[i][j]):
            tr_pairs += [[seqs[i], seqs[j]]]
            tr_y.append(incomplete_matrix[i][j])

    #tr_pairs = np.array(tr_pairs)
    #te_pairs = np.array(te_pairs)

    # network definition
    input_shape = (time_dim, feat_dim)
    base_network = create_base_network(input_shape, pad_value)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    dot = Lambda(batch_dot, output_shape=(1,))([processed_a, processed_b])

    model = Model([input_a, input_b], dot)

    # train
    rms = RMSprop(clipnorm=1.)
    model.compile(loss='mse', optimizer=rms)
    model_checkpoint = ModelCheckpoint('model.hdf5', save_best_only=True, save_weights_only=True)
    early_stopping = EarlyStopping(patience=15)
    history = History()
    tr_y = np.array(tr_y).astype('float32')
    # need to pad train data nad validation data

    model.fit([np.array(tr_pairs)[:, 0, :, :],
               np.array(tr_pairs)[:, 1, :, :]],
              tr_y,
              batch_size=128,
              epochs=1, # 3 is enough for test, 300 would be proper for actual usage
              callbacks=[model_checkpoint, early_stopping, history],
              validation_split=0.1,
              shuffle=True)

    # need to pad test data
    # compute final result on test set
    #print(model.evaluate([te_pairs[:, 0, :, :], te_pairs[:, 1, :, :]], te_y))
    preds = model.predict([np.array(te_pairs)[:, 0, :, :],
                           np.array(te_pairs)[:, 1, :, :]])
    print(preds)

    completed_matrix = incomplete_matrix.tolist()
    for k in range(te_pairs_index.__len__()):
        pred = preds[k]
        i, j = te_pairs_index[k]
        assert np.isnan(completed_matrix[i][j])
        completed_matrix[i][j] = pred
        assert not np.isnan(completed_matrix[i][j])
    assert not np.any(np.isnan(np.array(completed_matrix)))
    assert not np.any(np.isinf(np.array(completed_matrix)))
    return completed_matrix

def plot(file_name, similarities, files):
    # To fix the direction of the matrix as the diagonal line is from top-left to bottom-right.
    similarities_ = similarities[::-1]
    files_to_show = []
    for f in files:
        files_to_show.append(f.split('/')[-1].split('.')[0])
    files_to_show_ = files_to_show[::-1]
    
    trace = pgo.Heatmap(z=similarities_,
                        x=files_to_show,
                        y=files_to_show_,
                        zmin=0, zmax=1
    )
    data=[trace]
    po.plot(data, filename=file_name, auto_open=False)

def nearest_positive_semidefinite(matrix):
    sym_matrix = (matrix + matrix.T) * 0.5
    w, v = np.linalg.eig(sym_matrix)
    psd_w = np.array([max(0, e) for e in w])
    psd_matrix = np.dot(v, np.dot(np.diag(psd_w), np.linalg.inv(v)))
    return np.real(psd_matrix)
    
if __name__ == "__main__":
    filename = sys.argv[1]
    incomplete_percentage = int(sys.argv[2])
    mat = io.loadmat(filename)
    similarities = mat['gram']
    files = mat['indices']
    seqs = {}
    for f in files:
        #print(f)
        m = io.loadmat(f)
        seqs[f] = m['gest'].T

    random.seed(1)
        
    incomplete_similarities = []
    dropped_elements = []
    for i in range(similarities.__len__()):
        s_row = similarities[i]
        is_row = []
        for j in range(s_row.__len__()):
            s = s_row[j]
            if s == 1:
                if i == j:
                    is_row.append(1)
                    continue
            if random.randint(0, 99) < incomplete_percentage:
                is_row.append(np.nan)
                dropped_elements.append((i, j))
            else:
                is_row.append(s)
        incomplete_similarities.append(is_row)
    print("Incomplete matrix is provided.")

        
    completed_similarities = np.array(rnn_matrix_completion(incomplete_similarities, seqs, files))
    #assert np.nan not in completed_similarities.reshape([files.__len__() * files.__len__()])
    #assert np.inf not in completed_similarities.reshape([files.__len__() * files.__len__()])
    psd_completed_similarities = nearest_positive_semidefinite(completed_similarities)

    html_out_rnn = "./rnn_completion_test.html"
    mat_out_rnn = "./rnn_completion_test.mat"
    
    # "RnnCompletion"
    plot(html_out_rnn,
         psd_completed_similarities, files)
    io.savemat(mat_out_rnn, dict(gram=psd_completed_similarities, indices=files))
    print("RnnCompletion files are output.")

    mse = mean_squared_error(similarities, psd_completed_similarities)
    print("Mean squared error: " + str(mse))
    msede = mean_squared_error_of_dropped_elements(similarities, psd_completed_similarities, dropped_elements)
    print("Mean squared error of dropped elements: " + str(msede))
