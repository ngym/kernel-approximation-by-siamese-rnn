import copy, time

import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, SimpleRNN, LSTM, GRU, Masking, Activation, BatchNormalization
from keras.optimizers import Adam
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import Concatenate

from utils import multi_gpu


class SiameseRnn:
    def __init__(self, input_shape, pad_value, rnn_units, dense_units,
                 rnn, dropout, implementation, bidirectional, batchnormalization):
        """

        :param input_shape: Keras input shape
        :param pad_value: Padding value to be skipped among time steps
        :param rnn_units: Recurrent layer sizes
        :param dense_units: Dense layer sizes
        :param rnn: Recurrent Layer type (Vanilla, LSTM or GRU)
        :param dropout: Dropout probability
        :param implementation: RNN implementation (0: CPU, 2: GPU, 1: any)
        :param bidirectional: Flag to switch between Forward and Bidirectional RNN
        :param batchnormalization: Flag to switch Batch Normalization on/off
        :type input_shape: tuple
        :type pad_value: float
        :type rnn_units: list of ints
        :type dense_units: list of ints
        :type rnn: str
        :type dropout: float
        :type implementation: int
        :type bidirectional: bool
        :type batchnormalization: bool
        """
        self.input_shape = input_shape
        self.pad_value = pad_value
        self.rnn_units = rnn_units
        self.dense_units = dense_units
        self.rnn = rnn
        self.dropout = dropout
        self.implementation = implementation
        self.bidirectional = bidirectional
        self.batchnormalization = batchnormalization

        self.gpu_count = len(multi_gpu.get_available_gpus())
        self.model = self.__create_RNN_siamese_network()

    def __create_RNN_siamese_network(self):
        base_network = self.__create_RNN_base_network()
        input_a = Input(shape=self.input_shape)
        input_b = Input(shape=self.input_shape)
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)
        con = Concatenate()([processed_a, processed_b])
        parent = Dense(units=1, use_bias=False if self.batchnormalization else True)(con)
        if self.batchnormalization:
            parent = BatchNormalization()(parent)
        out = Activation('sigmoid')(parent)

        model = Model([input_a, input_b], out)

        optimizer = Adam(clipnorm=1.)
        if self.gpu_count > 1:
            model = multi_gpu.make_parallel(model, self.gpu_count)
        model.compile(loss='mse', optimizer=optimizer)

        return model

    def __create_RNN_base_network(self):
        """Keras Deep RNN network to be used as Siamese branch.
        Stacks some Recurrent and some Dense layers on top of each other

        :returns: Keras Deep RNN network
        :rtype: keras.engine.training.Model
        """
        seq = Sequential()
        seq.add(Masking(mask_value=self.pad_value, input_shape=self.input_shape, name="base_masking"))

        if self.rnn == "Vanilla":
            r = SimpleRNN
        elif self.rnn == "LSTM":
            r = LSTM
        elif self.rnn == "GRU":
            r = GRU
        else:
            raise NotImplementedError("Currently rnn must be Vanilla, LSTM or GRU!")

        if self.bidirectional:
            b = Bidirectional
        else:
            b = lambda x: x

        for i in range(len(self.rnn_units)):
            rnn_unit = self.rnn_units[i]
            return_sequences = (i < (len(self.rnn_units) - 1))
            seq.add(b(r(rnn_unit,
                        dropout=self.dropout, implementation=self.implementation,
                        return_sequences=return_sequences)))
            if self.batchnormalization and return_sequences:
                seq.add(BatchNormalization())
        for i in range(len(self.dense_units)):
            dense_unit = self.dense_units[i]
            seq.add(Dense(dense_unit, use_bias=False if self.batchnormalization else True))
            if self.batchnormalization:
                seq.add(BatchNormalization())
            if i < len(self.dense_units) - 1:
                seq.add(Activation('relu'))
            else:
                seq.add(Activation('relu', name='base_output'))
        return seq

    def train_and_validate(self, tr_indices, val_indices,
                           gram_drop,
                           seqs,
                           epochs,
                           patience,
                           logfile_loss,
                           logfile_hdf5):
        """Keras Siamese RNN training function.
        Carries out training and validation for given data over given number of epochs
        Logs results and network parameters

        :param tr_indices: Training 2-tuples of time series index pairs
        :param val_indices: Validation 2-tuples of time series index pairs
        :param gram_drop: Gram matrix with dropped elements
        :param seqs: List of time series
        :param epochs: Number of passes over data set
        :param patience: Early Stopping parameter
        :param logfile_loss: Log file name for results
        :param logfile_hdf5: Log file name for network structure and weights in HDF5 format
        :type tr_indices: list of tuples
        :type val_indices: list of tuples
        :type gram_drop: list of lists
        :type seqs: list of np.ndarrays
        :type epochs: int
        :type patience: int
        :type logfile_loss: str
        :type logfile_hdf5: str
        """

        def do_epoch(action, current_epoch, epoch_count,
                     indices, gram_drop, seqs, log_file):
            processed_sample_count = 0
            average_loss = 0
            if action == "training":
                np.random.shuffle(indices)
            gen = self.__generator_sequence_pairs(indices, gram_drop, seqs)
            start = curr_time = time.time()
            current_batch_iteration = 0
            while processed_sample_count < len(indices):
                # training batch
                x, y = next(gen)
                batch_loss = self.model.train_on_batch(x, y)
                average_loss = (average_loss * processed_sample_count + batch_loss * y.shape[0]) / \
                               (processed_sample_count + y.shape[0])
                processed_sample_count += y.shape[0]
                prev_time = curr_time
                curr_time = time.time()
                elapsed_time = curr_time - start
                eta = ((curr_time - prev_time) * len(indices) / y.shape[0]) - elapsed_time
                print_current_status(action, current_epoch, epoch_count,
                                     processed_sample_count, len(indices),
                                     elapsed_time, eta,
                                     average_loss, batch_loss,
                                     end='\r')
                log_current_status(log_file, action, current_epoch, current_batch_iteration, average_loss, batch_loss)
                current_batch_iteration += 1
            print_current_status(action, current_epoch, epoch_count,
                                 processed_sample_count, len(indices),
                                 elapsed_time, eta,
                                 average_loss, batch_loss)
            return average_loss

        def log_current_status(file, action, current_epoch, batch_iteration, average_loss, batch_loss):
            if action == "training":
                text = "%d, %d, %.5f, %.5f, nan, nan\n"
            else:
                text = "%d, %d, nan, nan, %.5f, %.5f\n"
            file.write(text %
                       (current_epoch, batch_iteration,
                        average_loss, batch_loss))
            file.flush()

        def print_current_status(action, current_epoch, epoch_count,
                                 processed_sample_count, total_sample_count,
                                 elapsed_time, eta, average_loss, loss_batch,
                                 end='\n'):
            print("epoch:[%d/%d] %s:[%d/%d] %ds, ETA:%ds, ave_loss:%.5f, loss_batch:%.5f" %
                  (current_epoch, epoch_count,
                   action, processed_sample_count, total_sample_count,
                   elapsed_time, eta,
                   average_loss, loss_batch),
                  end=end)

        loss_file = open(logfile_loss, "w")
        wait = 0
        best_validation_loss = np.inf
        loss_file.write("epoch, batch_iteration, average_training_loss, training_batch_loss, "
                        "average_validation_loss, validation_batch_loss\n")
        for epoch in range(1, epochs + 1):
            # training
            _ = do_epoch("training", epoch, epochs,
                         tr_indices, gram_drop, seqs, loss_file)

            # validation
            average_validation_loss = do_epoch("validation", epoch, epochs,
                                               val_indices, gram_drop, seqs, loss_file)

            if average_validation_loss < best_validation_loss:
                best_validation_loss = average_validation_loss
                self.model.save_weights(logfile_hdf5)
                best_weights = self.model.get_weights()
                wait = 0
            else:
                if wait >= patience:
                    self.model.set_weights(best_weights)
                    break
                wait += 1
        loss_file.close()

    def predict(self, te_indices, gram_drop, seqs):
        """Keras Siamese RNN prediction function.
        Carries out predicting for given data
        Logs results and network parameters

        :param te_indices: Testing 2-tuples of time series index pairs
        :param gram_drop: Gram matrix with dropped elements
        :param seqs: List of time series
        :type te_indices: list of tuples
        :type gram_drop: list of lists
        :type seqs: list of np.ndarrays
        :returns: List of predicted network outputs
        :rtype: list of lists
        """

        te_gen = self.__generator_sequence_pairs(te_indices, gram_drop, seqs)
        preds = []
        num_predicted_samples = 0
        while num_predicted_samples < len(te_indices):
            x, _ = next(te_gen)
            preds_batch = self.model.predict_on_batch(x)
            preds += preds_batch.tolist()
            num_predicted_samples += preds_batch.shape[0]
        return preds

    def __generator_sequence_pairs(self, indices, gram_drop, seqs):
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
        batch_size = 512 * self.gpu_count
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

    def load_weights(self, logfile_hdf5):
        self.model.load_weights(logfile_hdf5)
