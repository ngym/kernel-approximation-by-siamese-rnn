import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, SimpleRNN, LSTM, GRU, Masking, Activation, BatchNormalization, Lambda

from algorithms.siamese_rnn import SiameseRnn

class SiameseRnnBranch():
    def __init__(self, input_shape, pad_value, rnn_units, dense_units,
                 rnn, dropout, implementation, bidirectional, batchnormalization,
                 loss_function, siamese_joint_method,
                 trained_modelfile_hdf5,
                 siamese_arms_activation='linear'):
        siamese_model = SiameseRNN(input_shape, pad_value, rnn_units, dense_units,
                                   rnn, dropout, implementation, bidirectional, batchnormalization,
                                   loss_function, siamese_joint_method, siamese_arms_activation='linear')
        
        """
        print(K.get_value(model.layers[2].layers[1].weights[0])[0])
        model.load_weights()
        print(K.get_value(model.layers[2].layers[1].weights[0])[0])
        model = Model(input_a, processed_a)
        print(K.get_value(model.layers[1].layers[1].weights[0])[0])
        """
        print(K.get_value(siamese_model.layers[2].layers[1].weights[0])[0])
        siamese_model.load_weights(trained_modelfile_hdf5)
        print(K.get_value(siamese_model.layers[2].layers[1].weights[0])[0])
        model = Model(siamese_model.input_a, siamese_model.processed_a)
        print(K.get_value(model.layers[1].layers[1].weights[0])[0])
        model.compile(loss=loss_function, loss_weights=loss_weights, optimizer=optimizer)
         
        self.model = model
    def predict(self, seqs):
        train_feature = self.model.predict(seqs)
        return train_feature

        
