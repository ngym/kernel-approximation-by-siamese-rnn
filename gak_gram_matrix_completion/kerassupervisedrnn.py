"""Keras with Supervised Deep Recurrent Neural Networks.
"""

import time
import os
import numbers
results_dir = 'results/' + str(os.getpid())
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
os.environ['THEANO_FLAGS'] = 'base_compiledir=' + results_dir
from keras.models import Model
from keras.layers import *
from keras.layers.recurrent import _time_distributed_dense
from keras.initializers import Constant
from keras.optimizers import *
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping, History
from keras import backend as K
from algorithm import Algorithm

class ResidualRNN(SimpleRNN):
    """Fully-connected RNN where the output is to be fed back to input.
    """

    def __init__(self,
                 units,
                 hidden_units,
                 activation='relu',
                 encoder_activation='relu',
                 use_scale=True,
                 kernel_initializer='he_normal',
                 recurrent_initializer='he_normal',
                 decoder_initializer='he_normal',
                 decoder_regularizer=None,
                 decoder_constraint=None,
                 decoder_dropout=0.,
                 normalization_axes=-1,
                 momentum=0.99,
                 epsilon=0.001,
                 scale_initializer=Constant(0.1),#'ones',
                 moving_mean_initializer='zeros',
                 moving_var_initializer='ones',
                 scale_regularizer=None,
                 scale_constraint=None,
                 **kwargs):
        super(ResidualRNN, self).__init__(units, activation=activation, kernel_initializer=kernel_initializer,
                                          recurrent_initializer=recurrent_initializer, **kwargs)
        if self.stateful is True:
            raise NotImplementedError('Stateful mode is not implemented yet!')
        self.hidden_units = hidden_units
        self.encoder_activation = activations.get(encoder_activation)
        self.use_scale = use_scale
        self.decoder_initializer = initializers.get(decoder_initializer)
        self.decoder_regularizer = regularizers.get(decoder_regularizer)
        self.decoder_constraint = constraints.get(decoder_constraint)
        self.decoder_dropout = min(1., max(0., decoder_dropout))
        assert(normalization_axes in [-1, 2, [1, 2]])
        self.normalization_axes = normalization_axes
        self.momentum = momentum
        self.epsilon = epsilon
        self.scale_initializer = initializers.get(scale_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_var_initializer = initializers.get(moving_var_initializer)
        self.scale_regularizer = regularizers.get(scale_regularizer)
        self.scale_constraint = constraints.get(scale_constraint)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.time_dim = input_shape[1]
        self.input_dim = input_shape[2]
        self.input_spec = InputSpec(shape=(batch_size, None, self.input_dim))
        self.state_spec = InputSpec(shape=(batch_size, self.units))

        self.states = [None, None]
        if self.stateful:
            self.reset_states()

        self.kernel = self.add_weight((self.input_dim, self.hidden_units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight((self.units, self.hidden_units),
                                                initializer=self.recurrent_initializer,
                                                name='recurrent_kernel',
                                                regularizer=self.recurrent_regularizer,
                                                constraint=self.recurrent_constraint)
        self.decoder_kernel = self.add_weight((self.hidden_units, self.units),
                                              initializer=self.decoder_initializer,
                                              name='decoder_kernel',
                                              regularizer=self.decoder_regularizer,
                                              constraint=self.decoder_constraint)

        if self.use_scale:
            self.kernel_scale = self.add_weight((self.hidden_units,),
                                                initializer=self.scale_initializer,
                                                name='kernel_scale',
                                                regularizer=self.scale_regularizer,
                                                constraint=self.scale_constraint)
            self.recurrent_scale = self.add_weight((self.hidden_units,),
                                                   initializer=self.scale_initializer,
                                                   name='recurrent_scale',
                                                   regularizer=self.scale_regularizer,
                                                   constraint=self.scale_constraint)
            self.decoder_scale = self.add_weight((self.units,),
                                                 initializer=self.scale_initializer,
                                                 name='decoder_scale',
                                                 regularizer=self.scale_regularizer,
                                                 constraint=self.scale_constraint)
        else:
            self.kernel_scale = None
            self.recurrent_scale = None
            self.decoder_scale = None

        if self.use_bias:
            self.encoder_bias = self.add_weight((self.hidden_units,),
                                        name='encoder_bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.decoder_bias = self.add_weight((self.units,),
                                                initializer=self.bias_initializer,
                                                name='decoder_bias',
                                                regularizer=self.bias_regularizer,
                                                constraint=self.bias_constraint)
        else:
            self.encoder_bias = None
            self.decoder_bias = None

        if self.normalization_axes in [-1, 2]:
            self.encoder_shape = (self.hidden_units,)
            self.decoder_shape = (self.units,)
        elif self.normalization_axes == [1, 2]:
            self.encoder_shape = (self.time_dim, self.hidden_units,)
            self.decoder_shape = (self.time_dim, self.units,)

        self.kernel_moving_mean = self.add_weight(self.encoder_shape,
                                                  initializer=self.moving_mean_initializer,
                                                  name='kernel_moving_mean',
                                                  trainable=False)
        self.kernel_moving_var = self.add_weight(self.encoder_shape,
                                                 initializer=self.moving_var_initializer,
                                                 name='kernel_moving_var',
                                                 trainable=False)
        self.recurrent_moving_mean = self.add_weight(self.encoder_shape,
                                                     initializer=self.moving_mean_initializer,
                                                     name='recurrent_moving_mean',
                                                     trainable=False)
        self.recurrent_moving_var = self.add_weight(self.encoder_shape,
                                                    initializer=self.moving_var_initializer,
                                                    name='recurrent_moving_var',
                                                    trainable=False)
        self.decoder_moving_mean = self.add_weight(self.decoder_shape,
                                                   initializer=self.moving_mean_initializer,
                                                   name='decoder_moving_mean',
                                                   trainable=False)
        self.decoder_moving_var = self.add_weight(self.decoder_shape,
                                                  initializer=self.moving_var_initializer,
                                                  name='decoder_moving_var',
                                                  trainable=False)

    def get_initial_state(self, inputs):
        initial_state = super(ResidualRNN, self).get_initial_state(inputs)
        initial_state[0] = K.zeros((initial_state[0].shape[0], 3 * self.units), dtype='float32')
        initial_state[1] = K.expand_dims(K.zeros_like(initial_state[1][:, 0], dtype='int32'), axis=1)
        return initial_state

    def preprocess_input(self, inputs, training=None):
        if self.implementation > 0:
            return inputs
        else:
            input_shape = K.int_shape(inputs)
            input_dim = input_shape[2]
            timesteps = input_shape[1]
            return _time_distributed_dense(inputs,
                                           self.kernel,
                                           K.zeros((self.hidden_units,)),
                                           self.dropout,
                                           input_dim,
                                           self.hidden_units,
                                           timesteps,
                                           training=training)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        # input shape: `(samples, time (padded with zeros), input_dim)`
        # note that the .build() method of subclasses MUST define
        # self.input_spec and self.state_spec with complete input shapes.
        if isinstance(inputs, list):
            initial_state = inputs[1:]
            inputs = inputs[0]
        elif initial_state is not None:
            pass
        elif self.stateful:
            initial_state = self.states
        else:
            initial_state = self.get_initial_state(inputs)

        if isinstance(mask, list):
            mask = mask[0]

        if len(initial_state) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) +
                             ' states but was passed ' +
                             str(len(initial_state)) +
                             ' initial states.')
        input_shape = K.int_shape(inputs)
        if self.unroll and input_shape[1] is None:
            raise ValueError('Cannot unroll a RNN if the '
                             'time dimension is undefined. \n'
                             '- If using a Sequential model, '
                             'specify the time dimension by passing '
                             'an `input_shape` or `batch_input_shape` '
                             'argument to your first layer. If your '
                             'first layer is an Embedding, you can '
                             'also use the `input_length` argument.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a `shape` '
                             'or `batch_shape` argument to your Input layer.')
        constants = self.get_constants(inputs, training=None)
        preprocessed_input = self.preprocess_input(inputs, training=None)
        last_output, outputs, states = K.rnn(lambda i, s: self.step(i, s, training=training),
                                             preprocessed_input,
                                             initial_state,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])

        outputs, h, u, v = outputs[:, :, :self.units], outputs[:, :, self.units:(self.units + self.hidden_units)], \
                           outputs[:, :, (self.units + self.hidden_units):(self.units + 2 * self.hidden_units)], \
                           outputs[:, :, (self.units + 2 * self.hidden_units):]
        last_output = last_output[:, :self.units]

        if self.normalization_axes in [-1, 2]:
            reduction_axes = [0, 1]
        elif self.normalization_axes == [1, 2]:
            reduction_axes = [0]
        h_mean, h_var = h.mean(reduction_axes), h.var(reduction_axes)
        u_mean, u_var = u.mean(reduction_axes), u.var(reduction_axes)
        v_mean, v_var = v.mean(reduction_axes), v.var(reduction_axes)

        self.add_update([K.moving_average_update(self.kernel_moving_mean,
                                                 K.in_train_phase(h_mean, self.kernel_moving_mean, training=training),
                                                 self.momentum),
                         K.moving_average_update(self.kernel_moving_var,
                                                 K.in_train_phase(h_var, self.kernel_moving_var, training=training),
                                                 self.momentum),
                         K.moving_average_update(self.recurrent_moving_mean,
                                                 K.in_train_phase(u_mean, self.recurrent_moving_mean,
                                                                  training=training),
                                                 self.momentum),
                         K.moving_average_update(self.recurrent_moving_var,
                                                 K.in_train_phase(u_var, self.recurrent_moving_var,
                                                                  training=training),
                                                 self.momentum),
                         K.moving_average_update(self.decoder_moving_mean,
                                                 K.in_train_phase(v_mean, self.decoder_moving_mean,
                                                                  training=training),
                                                 self.momentum),
                         K.moving_average_update(self.decoder_moving_var,
                                                 K.in_train_phase(v_var, self.decoder_moving_var,
                                                                  training=training),
                                                 self.momentum)],
                        inputs)

        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, inputs)

        # Properly set learning phase
        last_output._uses_learning_phase = True
        outputs._uses_learning_phase = True

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def step(self, inputs, states, training=None):
        prev_output = states[0]
        t = states[1]
        B_kernel = states[-3]
        B_recurrent = states[-2]
        B_decoder = states[-1]


        prev_output = prev_output[:, :self.units]
        if self.normalization_axes in [-1, 2]:
            kernel_moving_mean = self.kernel_moving_mean
            kernel_moving_var = self.kernel_moving_var
            recurrent_moving_mean = self.recurrent_moving_mean
            recurrent_moving_var = self.recurrent_moving_var
            decoder_moving_mean = self.decoder_moving_mean
            decoder_moving_var = self.decoder_moving_var
        elif self.normalization_axes == [1, 2]:
            kernel_moving_mean = self.kernel_moving_mean[t[0, 0], :]
            kernel_moving_var = self.kernel_moving_var[t[0, 0], :]
            recurrent_moving_mean = self.recurrent_moving_mean[t[0, 0], :]
            recurrent_moving_var = self.recurrent_moving_var[t[0, 0], :]
            decoder_moving_mean = self.decoder_moving_mean[t[0, 0], :]
            decoder_moving_var = self.decoder_moving_var[t[0, 0], :]

        if self.implementation == 0:
            h = inputs
        else:
            if 0 < self.dropout < 1:
                h = K.dot(inputs * B_kernel, self.kernel)
            else:
                h = K.dot(inputs, self.kernel)
        bn_h_train, _, _ = K.normalize_batch_in_training(h, self.kernel_scale, None, [0], epsilon=self.epsilon)
        bn_h_test = K.batch_normalization(h, K.expand_dims(kernel_moving_mean, axis=0),
                                          K.expand_dims(kernel_moving_var, axis=0), None,
                                          K.expand_dims(self.kernel_scale, axis=0) if self.kernel_scale is not None
                                          else None, epsilon=self.epsilon)
        bn_h = K.in_train_phase(bn_h_train, bn_h_test, training=training)

        if 0 < self.recurrent_dropout < 1:
            prev_output *= B_recurrent
        u = K.dot(prev_output, self.recurrent_kernel)
        bn_u_train, _, _ = K.normalize_batch_in_training(u, self.recurrent_scale, None, [0],
                                                         epsilon=self.epsilon)
        bn_u_test = K.batch_normalization(u, K.expand_dims(recurrent_moving_mean, axis=0),
                                          K.expand_dims(recurrent_moving_var, axis=0), None,
                                          K.expand_dims(self.recurrent_scale, axis=0)
                                          if self.recurrent_scale is not None else None, epsilon=self.epsilon)
        bn_u = K.in_train_phase(bn_u_train, bn_u_test, training=training)

        output_encoder = bn_h + bn_u
        if self.encoder_bias is not None:
            output_encoder = K.bias_add(output_encoder, self.encoder_bias)
        if self.encoder_activation is not None:
            output_encoder = self.encoder_activation(output_encoder)

        if 0 < self.decoder_dropout < 1:
            output_encoder *= B_decoder
        v = K.dot(output_encoder, self.decoder_kernel)
        bn_v_train, _, _ = K.normalize_batch_in_training(v, self.decoder_scale, None, [0], epsilon=self.epsilon)
        bn_v_test = K.batch_normalization(v, K.expand_dims(decoder_moving_mean, axis=0),
                                          K.expand_dims(decoder_moving_var, axis=0), None,
                                          K.expand_dims(self.decoder_scale, axis=0)
                                          if self.decoder_scale is not None else None, epsilon=self.epsilon)
        bn_v = K.in_train_phase(bn_v_train, bn_v_test, training=training)

        output = prev_output + bn_v
        if self.decoder_bias is not None:
            output = K.bias_add(output, self.decoder_bias)
        if self.activation is not None:
            output = self.activation(output)

        # Properly set learning phase on output tensor.
        output._uses_learning_phase = True
        concat = K.concatenate([output, h, u, v], axis=1)
        concat._uses_learning_phase = True

        t = t + 1

        return concat, [concat, t]

    def get_constants(self, inputs, training=None):
        constants = super(ResidualRNN, self).get_constants(inputs, training=training)

        if 0 < self.decoder_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.hidden_units))

            def dropped_inputs():
                return K.dropout(ones, self.decoder_dropout)

            dec_dp_mask = K.in_train_phase(dropped_inputs,
                                           ones,
                                           training=training)
            constants.append(dec_dp_mask)
        else:
            constants.append(K.cast_to_floatx(1.))

        return constants

    def get_config(self):
        config = {'hidden_units': self.hidden_units,
                  'encoder_activation': activations.serialize(self.encoder_activation),
                  'use_scale': self.use_scale,
                  'decoder_initializer': initializers.serialize(self.decoder_initializer),
                  'decoder_regularizer': regularizers.serialize(self.decoder_regularizer),
                  'decoder_constraint': constraints.serialize(self.decoder_constraint),
                  'decoder_dropout': self.decoder_dropout,
                  'normalization_axes': self.normalization_axes,
                  'momentum': self.momentum,
                  'epsilon': self.epsilon,
                  'scale_initializer': initializers.serialize(self.scale_initializer),
                  'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
                  'moving_var_initializer': initializers.serialize(self.moving_var_initializer),
                  'scale_regularizer': regularizers.serialize(self.scale_regularizer),
                  'scale_constraint': constraints.serialize(self.scale_constraint)}
        base_config = super(ResidualRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class KerasSupervisedRNN(Algorithm):
    """Solve multivariate total variation minimization via Keras with Supervised Deep Recurrent Neural Networks.
    
    :math:`\min_{\\theta} L(\\theta, X, Y) = - 1/R \sum_{r=1}^{R} \sum_{c=1}^{C} Y_{r,c} \log{f_{\\theta}(X)_{r,c}}`.
    
    Minimize categorical cross-entropy between predictions and targets.

    :param hyperparams: Hyperparameter configuration
    :key 'shape2': Number of input features (int)
    :key 'rnn': Number of active recurrent layers (int)
    :key 'rnn1_units': Number of units in recurrent layer 1 (int)
    :key 'rnn1_dropout': Dropout probability for recurrent layer 1 (float)
    :key 'rnn1_recurrent_dropout': Recurrent dropout probability for recurrent layer 1 (float)
    :key 'rnn2_units': Number of units in recurrent layer 1 (int)
    :key 'rnn2_dropout': Dropout probability for recurrent layer 2 (float)
    :key 'rnn2_recurrent_dropout': Recurrent dropout probability for recurrent layer 2 (float)
    :key 'bidirectional': Whether to use bidirectional or forward recurrent layers (bool)
    :key 'lstm': Whether to use lstm or vanilla recurrent layers (bool)
    :key 'l2': Weight regularization parameter (float)
    :key 'lr': Learning rate (float)
    :key 'rho': RMSprop parameter (float)
    :key 'epsilon': Fuzz factor (float)
    :key 'decay': Learning rate decay (float)
    :key 'clipnorm': Gradient norm clipping parameter (float)
    :key 'patience': Early stopping patience (int)
    :key 'batch_size': Number of samples per gradient update (int)
    :key 'epochs': Number of times to iterate over the training data (int) 
    :type hyperparams: dict

    :Example:

    >>> import numpy as np
    >>> from algorithms.kerassupervisedrnn import KerasSupervisedRNN
    >>> alg = KerasSupervisedRNN()
    >>> alg.set_hyperparams({'shape2': 2})
    >>> alg.predict({'X': np.zeros((1, 3, 2)), 'W': np.ones((1, 3, 2))})['Y_hat'][0, :]
    np.zeros((3,))

    References:
        [1] `Sutskever, Ilya. "Training recurrent neural networks." 
        <https://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf>`_.
        
        [2] `Graves, Alex, Abdel-rahman Mohamed, and Geoffrey Hinton. "Speech recognition with deep recurrent neural 
        networks." <https://arxiv.org/pdf/1303.5778.pdf)?>`_.
        
        [3] `Schuster, Mike, and Kuldip K. Paliwal. "Bidirectional recurrent neural networks."
        <https://www.researchgate.net/profile/Mike_Schuster/publication/3316656_Bidirectional_recurrent_neural_networks/
        links/56861d4008ae19758395f85c.pdf>`_.
        
        [4] `Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory." 
        <http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf>`_.        
    """

    def __init__(self, hyperparams={'shape1': 150, 'shape2': 2, 'shape_out': 1, 'rnn': 2, 'rnn1_units': 256,
                                    'rnn1_hidden_units': 256, 'rnn1_dropout': 0., 'rnn1_recurrent_dropout': 0.,
                                    'rnn1_decoder_dropout': 0., 'rnn2_units': 256, 'rnn2_hidden_units': 256,
                                    'rnn2_dropout': 0., 'rnn2_recurrent_dropout': 0., 'rnn2_decoder_dropout': 0.,
                                    'return_sequences': False, 'activation': 'softmax', 'bidirectional': False,
                                    'mode': 'lstm', 'l2': 0.01, 'lr': 0.001, 'rho': 0.9, 'epsilon': 1e-8, 'decay': 0.,
                                    'clipnorm': 1., 'patience': 20, 'batch_size': 64, 'epochs': 1024}):
        super(KerasSupervisedRNN, self).__init__(hyperparams=hyperparams)
    
    def build(self):
        network = dict()
        network['inp'] = Input(shape=(self.hyperparams['shape1'], self.hyperparams['shape2']), dtype='float32')
        network['mask'] = Masking(mask_value=-123456789)(network['inp'])
        parent = network['mask']
        if self.hyperparams['bidirectional'] is True:
            bidir = Bidirectional
        else:
            def bidir(x):
                return x
        for k in range(1, self.hyperparams['rnn'] + 1):
            if self.hyperparams['mode'] is 'lstm':
                rnn = LSTM(units=self.hyperparams['rnn' + str(k) + '_units'],
                           dropout=self.hyperparams['rnn' + str(k) + '_dropout'],
                           recurrent_dropout=self.hyperparams['rnn' + str(k) + '_recurrent_dropout'],
                           kernel_regularizer=l2(self.hyperparams['l2']),
                           return_sequences=True)
            elif self.hyperparams['mode'] is 'vanilla':
                rnn = SimpleRNN(units=self.hyperparams['rnn' + str(k) + '_units'],
                                dropout=self.hyperparams['rnn' + str(k) + '_dropout'],
                                recurrent_dropout=self.hyperparams['rnn' + str(k) + '_recurrent_dropout'],
                                kernel_regularizer=l2(self.hyperparams['l2']),
                                return_sequences=True)
            elif self.hyperparams['mode'] is 'residual':
                rnn = ResidualRNN(units=self.hyperparams['rnn' + str(k) + '_units'],
                                  hidden_units=self.hyperparams['rnn' + str(k) + '_hidden_units'],
                                  dropout=self.hyperparams['rnn' + str(k) + '_dropout'],
                                  recurrent_dropout=self.hyperparams['rnn' + str(k) + '_recurrent_dropout'],
                                  decoder_dropout=self.hyperparams['rnn' + str(k) + '_decoder_dropout'],
                                  normalization_axes=[1, 2], kernel_regularizer=l2(self.hyperparams['l2']),
                                  recurrent_regularizer=l2(self.hyperparams['l2']),
                                  decoder_regularizer=l2(self.hyperparams['l2']), return_sequences=True)
            else:
                raise NotImplementedError('mode must be lstm, vanilla or residual')
            network['rnn' + str(k) + '_units'] = bidir(rnn)(parent)
            parent = network['rnn' + str(k) + '_units']
            # parent = BatchNormalization()(network['rnn' + str(k) + '_units'])
        network['dense'] = TimeDistributed(Dense(units=self.hyperparams['shape_out'],
                                                 activation=activations.get(self.hyperparams['activation']),
                                                 kernel_regularizer=l2(self.hyperparams['l2'])))(parent)
        network['out'] = Lambda(lambda x: x[:, -1, :],
                                output_shape=(self.hyperparams['shape_out'],))(network['dense']) \
            if self.hyperparams['return_sequences'] is False else network['dense']

        self.model = Model(inputs=network['inp'], outputs=network['out'])
        self.model.compile(loss='categorical_crossentropy' if self.hyperparams['activation'] is 'softmax' else 'mse',
                           optimizer=RMSprop(lr=self.hyperparams['lr'], rho=self.hyperparams['rho'],
                                             epsilon=self.hyperparams['epsilon'], decay=self.hyperparams['decay'],
                                             clipnorm=self.hyperparams['clipnorm']),
                           metrics=['categorical_accuracy'] if self.hyperparams['activation'] is 'softmax'
                           else [])

    def train(self, tensors_train, tensors_val, validate=True):
        self.check_tensors(tensors_train, self.input_ndims)
        self.check_tensors(tensors_val, self.input_ndims)
        hdf5 = results_dir + '/model.hdf5'
        if os.path.exists(hdf5):
            os.remove(hdf5)
        self.model.save_weights(hdf5)
        model_checkpoint = ModelCheckpoint(hdf5, save_best_only=True, save_weights_only=True)
        early_stopping = EarlyStopping(patience=self.hyperparams['patience'])
        history = History()
        try:
            self.model.fit(tensors_train['X'],
                           tensors_train['Y'],
                           batch_size=self.hyperparams['batch_size'],
                           epochs=self.hyperparams['epochs'],
                           verbose=1,  # 0,
                           callbacks=[model_checkpoint, early_stopping, history],
                           validation_data=(tensors_val['X'], tensors_val['Y']) if validate is True \
                               else (tensors_train['X'], tensors_train['Y']),
                           shuffle=True)
        except TimeoutError:
            pass
        if hasattr(history, 'history'):
            self.history = history.history
            if 'val_loss' in self.history.keys():
                self.hyperparams['epochs'] = int(np.argmin(self.history['val_loss']) + 1)
        else:
            self.history = {}
        self.model.load_weights(hdf5)
        return self.test(tensors_val)

    def test(self, tensors):
        self.check_tensors(tensors, self.input_ndims)
        time_start = time.time()
        losses = self.model.evaluate(tensors['X'], tensors['Y'], self.hyperparams['batch_size'])
        if isinstance(losses, numbers.Number):
            losses = [losses]
        time_delta = time.time() - time_start
        time_mean = time_delta / tensors['X'].shape[0]
        keras_result = {n: l for (n, l) in zip(self.model.metrics_names, losses)}
        hyperopt_result = {'loss': losses[0], 'time': time.time(), 'time_mean': time_mean,
                           'model_yaml': self.model.to_yaml(), 'model_json': self.model.to_json(),
                           'weights': self.model.get_weights(),
                           'history': self.history if hasattr(self, 'history') else None}
        return dict(list(keras_result.items()) + list(hyperopt_result.items()))

    def predict(self, tensors):
        self.check_tensors(tensors, self.input_ndims)
        time_start = time.time()
        Y_hat = self.model.predict(tensors['X'], self.hyperparams['batch_size'])
        '''
        if self.hyperparams['activation'] is 'softmax':
            Y_hat = np.argmax(Y_hat, axis=1)[:, None]
        '''
        time_delta = time.time() - time_start
        time_mean = time_delta / tensors['X'].shape[0]
        output = {'Y_hat': Y_hat, 'time_mean': np.asarray(time_mean)}
        self.check_tensors(output, self.output_ndims)
        return output
        
    def hyperopt_load_best_trial(self, filename=results_dir + '/trials.p'):
        best_trial = super(KerasSupervisedRNN, self).hyperopt_load_best_trial(filename)
        self.model.set_weights(best_trial.get('result').get('weights'))
        return best_trial

    def load_json(self, filename=results_dir + '/1/run.json'):
        jsn = super(KerasSupervisedRNN, self).load_json(filename)
        self.model.set_weights(jsn['result']['weights'])
        return jsn
