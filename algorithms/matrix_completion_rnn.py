import sys, copy, time, json
import numpy as np
from scipy import io

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, LSTM, Masking, Activation, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import Concatenate

from utils.nearest_positive_semidefinite import nearest_positive_semidefinite
from utils.errors import mean_squared_error, mean_absolute_error, relative_error
from utils.plot_gram_matrix import plot
from utils.make_matrix_incomplete import drop_gram_random, drop_samples
from datasets.find_and_read_sequences import find_and_read_sequences
from utils.multi_gpu import make_parallel


ngpus = 2

def create_LSTM_base_network(input_shape, mask_value,
                             lstm_units=[5], dense_units=[2],
                             dropout=0.3,
                             implementation=2, bidirectional=False):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(Masking(mask_value=mask_value, input_shape=input_shape))

    if bidirectional:
        f = Bidirectional
    else:
        f = lambda x: x

    for i in range(len(lstm_units)):
        lstm_unit = lstm_units[i]
        if i == len(lstm_units) - 1:
            return_sequences = False
        else:
            return_sequences = True
        seq.add(f(LSTM(lstm_unit,
                       dropout=dropout, implementation=implementation,
                       return_sequences=return_sequences)))

    for i in range(len(dense_units)):
        dense_unit = dense_units[i]
        if i == len(dense_units) - 1:
            activation='linear'
        else:
            activation='relu'
        seq.add(Dense(dense_unit, activation=activation))
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

def train_and_validate(model, tr_indices, v_indices,
                       incomplete_matrix,
                       seqs,
                       epochs,
                       patience,
                       lossesfile,
                       hdf5_out_rnn):
    fd_losses = open(lossesfile, "w")

    list_ave_tr_loss = []
    list_tr_loss_batch = []
    list_ave_v_loss = []
    list_v_loss_batch = []
    wait = 0
    best_v_loss = np.inf
    fd_losses.write("epoch, num_batch_iteration, ave_tr_loss, tr_loss_batch, ave_v_loss, v_loss_batch\n")
    for epoch in range(1, epochs + 1):
        num_trained_samples = 0
        ave_tr_loss = 0
        np.random.shuffle(tr_indices)
        tr_gen = generator_sequence_pairs(tr_indices, incomplete_matrix, seqs)
        tr_start = cur_time = time.time()
        num_batch_iteration = 0
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

        num_validated_samples = 0
        ave_v_loss = 0
        v_gen  = generator_sequence_pairs(v_indices, incomplete_matrix, seqs)
        v_start = cur_time = time.time()
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
                   len(v_indices), cur_time - v_start,
                   ((cur_time - prev_time) * len(v_indices) / y.shape[0]) - (cur_time - v_start),
                   ave_v_loss, v_loss_batch), end='\r')
            list_ave_v_loss.append(ave_v_loss)
            list_v_loss_batch.append(v_loss_batch)
            fd_losses.write("%d, %d, nan, nan, %.5f, %.5f\n" % (epoch, num_batch_iteration, ave_v_loss, v_loss_batch))
            fd_losses.flush()
            num_batch_iteration += 1
        print("                                                        epoch:[%d/%d] validation:[%d/%d] %ds, ETA:%ds, ave_loss:%.5f, loss_batch:%.5f" %
              (epoch, epochs, num_validated_samples,
               len(v_indices), cur_time - v_start,
               ((cur_time - prev_time) * len(v_indices) / y.shape[0]) - (cur_time - v_start),
               ave_v_loss, v_loss_batch))
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
    fd_losses.close()

def test(model, te_indices, incomplete_matrix, seqs):
    te_gen = generator_sequence_pairs(te_indices, incomplete_matrix, seqs)
    preds = []
    num_predicted_samples = 0
    while num_predicted_samples < len(te_indices):
        x, _ = next(te_gen)
        preds_batch = model.predict_on_batch(x)
        preds += preds_batch.tolist()
        num_predicted_samples += preds_batch.shape[0]
    return preds

def rnn_matrix_completion(incomplete_matrix_, seqs_,
                          epochs, patience,
                          lossesfile, hdf5_out_rnn,
                          rnn,
                          lstm_units, dense_units,
                          dropout,
                          implementation,
                          bidirectional):
    num_seqs = len(seqs_)
    incomplete_matrix = np.array(incomplete_matrix_)
    time_dim = max([seq_.shape[0] for seq_ in seqs_.values()])

    pad_value = -123456789 # np.inf #np.nan # 0 #np.NINF #np.inf
    seqs = pad_sequences([s.tolist() for s in seqs_.values()],
                         maxlen=time_dim, dtype='float32',
                         padding='post', value=pad_value)

    feat_dim = seqs[0].shape[1]

    # network definition
    K.clear_session()

    input_shape = (time_dim, feat_dim)
    if rnn == "LSTM":
        base_network = create_LSTM_base_network(input_shape, pad_value,
                                                lstm_units, dense_units,
                                                dropout,
                                                implementation,
                                                bidirectional)
    else:
        print("invalid RNN network.")
        assert False

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    con = Concatenate()([processed_a, processed_b])
    dns = Dense(units=1, activation='linear')(con)
    out = Activation('sigmoid')(dns)

    model = Model([input_a, input_b], out)

    # train
    rms = Adam(clipnorm=1.)

    if ngpus > 1:
        model = make_parallel(model, ngpus)
    model.compile(loss='mse', optimizer=rms)
    # need to pad train data nad validation data

    """
    generate random random permutation of (i,j)
    split 90 and 10
    make gentrain
    make genval
    """
    # for choose validation data at random
    tv_indices = np.random.permutation([(i, j)
                                        for i in range(num_seqs)
                                        for j in range(i, num_seqs)
                                        if not np.isnan(incomplete_matrix[i][j])])
    tr_indices = tv_indices[:int(len(tv_indices) * 0.9)]
    v_indices = tv_indices[int(len(tv_indices) * 0.9):]

    fit_start = time.time()
    train_and_validate(model, tr_indices, v_indices,
                       incomplete_matrix,
                       seqs,
                       epochs,
                       patience,
                       lossesfile,
                       hdf5_out_rnn)
    fit_finish = time.time()

    # need to pad test data
    # compute final result on test set
    #print(model.evaluate([te_pairs[:, 0, :, :], te_pairs[:, 1, :, :]], te_y))

    te_indices = [(i, j)
                  for i in range(num_seqs)
                  for j in range(i, num_seqs)
                  if np.isnan(incomplete_matrix[i][j])]
    #num_dropped = len(te_indices)

    pred_start = time.time()
    preds = test(model, te_indices, incomplete_matrix, seqs)
    pred_finish = time.time()

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

    return completed_matrix, fit_start, fit_finish, pred_start, pred_finish

def main():
    main_start = time.time()
    if len(sys.argv) != 2:
        random_drop = True
        gram_filename = sys.argv[1]
        incomplete_percentage = int(sys.argv[2])
        completionanalysisfile = sys.argv[3]
        epochs = 2
        patience = 2
        rnn = "LSTM"
        lstm_units = [5]
        dense_units = [2]
        dropout = 0.3
        implementation = 2
        bidirectional = False
    else:
        random_drop = False
        config_json_file = sys.argv[1]
        config_dict = json.load(open(config_json_file, 'r'))

        gram_filename = config_dict['gram_file']
        indices_to_drop = config_dict['indices_to_drop']
        completionanalysisfile = config_dict['completionanalysisfile']
        epochs = config_dict['epochs']
        patience = config_dict['patience']
        rnn = config_dict['rnn']
        lstm_units = config_dict['lstm_units']
        dense_units = config_dict['dense_units']
        dropout = config_dict['dropout']
        implementation = config_dict['implementation']
        bidirectional = config_dict['bidirectional']

    mat = io.loadmat(gram_filename)
    similarities = mat['gram']
    files = mat['indices']

    lossesfile = completionanalysisfile.replace(".timelog", ".losses")
    
    seqs = find_and_read_sequences(gram_filename, files)

    seed = 1

    if random_drop:
        incomplete_similarities, dropped_elements = drop_gram_random(seed, similarities, incomplete_percentage)
        html_out_rnn = gram_filename.replace(".mat", "_loss" + str(incomplete_percentage) + "_RNN_LSTM.html")
        mat_out_rnn  = gram_filename.replace(".mat", "_loss" + str(incomplete_percentage) + "_RNN_LSTM.mat")
        hdf5_out_rnn  = gram_filename.replace(".mat", "_loss" + str(incomplete_percentage) + "_RNN_LSTM.hdf5")
    else:
        incomplete_similarities, dropped_elements = drop_samples(similarities, indices_to_drop)
        html_out_rnn = gram_filename.replace(".mat", "_lossfrom" + str(indices_to_drop[0]) + "_RNN_LSTM.html")
        mat_out_rnn  = gram_filename.replace(".mat", "_lossfrom" + str(indices_to_drop[0]) + "_RNN_LSTM.mat")
        hdf5_out_rnn  = gram_filename.replace(".mat", "_lossfrom" + str(indices_to_drop[0]) + "_RNN_LSTM.hdf5")

    # "RnnCompletion"
    completed_similarities, fit_start, fit_finish, pred_start, \
        pred_finish = rnn_matrix_completion(incomplete_similarities, seqs,
                                            epochs, patience,
                                            lossesfile, hdf5_out_rnn,
                                            rnn,
                                            lstm_units, dense_units,
                                            dropout,
                                            implementation,
                                            bidirectional)
    completed_similarities = np.array(completed_similarities)
    # eigenvalue check
    npsd_start = time.time()
    psd_completed_similarities = nearest_positive_semidefinite(completed_similarities)
    npsd_finish = time.time()

    # OUTPUT
    plot(html_out_rnn,
         psd_completed_similarities, files)
    if random_drop:
        io.savemat(mat_out_rnn, dict(gram=psd_completed_similarities,
                                     non_psd_gram=completed_similarities,
                                     dropped_gram=incomplete_similarities,
                                     orig_gram=similarities,
                                     indices=files))
    else:
        io.savemat(mat_out_rnn, dict(gram=psd_completed_similarities,
                                     non_psd_gram=completed_similarities,
                                     dropped_gram=incomplete_similarities,
                                     dropped_indices_number=indices_to_drop,
                                     orig_gram=similarities,
                                     indices=files))

    print("RnnCompletion files are output.")

    mse = mean_squared_error(similarities, psd_completed_similarities)
    msede = mean_squared_error(similarities,
                               psd_completed_similarities,
                               dropped_elements)

    mae = mean_absolute_error(similarities, psd_completed_similarities)
    maede = mean_absolute_error(similarities,
                                psd_completed_similarities,
                                dropped_elements)
    
    re = relative_error(similarities,
                        psd_completed_similarities)
    rede = relative_error(similarities,
                         psd_completed_similarities,
                         dropped_elements)

    main_finish = time.time()
    
    analysis_json = {}
    analysis_json['number_of_dropped_elements'] = len(dropped_elements)
    analysis_json['fit_starts'] = fit_start
    analysis_json['fit_finishes'] = fit_finish
    analysis_json['fit_duration'] = fit_finish - fit_start
    analysis_json['pred_starts'] = pred_start
    analysis_json['pred_finishes'] = pred_finish
    analysis_json['pred_duration'] = pred_finish - pred_start
    analysis_json['npsd_start'] = npsd_start
    analysis_json['npsd_finish'] = npsd_finish
    analysis_json['npsd_duration'] = npsd_finish - npsd_start
    analysis_json['main_start'] = main_start
    analysis_json['main_finish'] = main_finish
    analysis_json['main_duration'] = main_finish - main_start
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
