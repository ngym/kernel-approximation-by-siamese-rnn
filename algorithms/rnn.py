import copy, time

import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, SimpleRNN, LSTM, GRU, Masking, Activation, BatchNormalization
from keras.optimizers import Adam
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import Concatenate

from utils import multi_gpu

class Rnn:
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

    def create_RNN_base_network(self):
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
                        return_sequences=return_sequences, activation='relu')))
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
                pass
                #seq.add(Activation('linear', name='base_output'))
        return seq

    def load_weights(self, logfile_hdf5):
        self.model.load_weights(logfile_hdf5)

        
