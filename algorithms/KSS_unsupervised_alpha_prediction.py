import copy, time

import numpy as np

import keras.backend as K
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.layers import Dense, Dropout, Input, SimpleRNN, LSTM, GRU, Masking, Activation, BatchNormalization
from keras.optimizers import Adam
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import Concatenate

import keras.initializers as initializers
import keras.regularizers as regularizers
import keras.constraints as constraints
from keras.legacy import interfaces
from keras.engine import Layer, InputSpec

import tensorflow as tf

from utils import multi_gpu
from rnn import Rnn
from algorithms.kernel_group_lasso import kernel_group_lasso

class unsupervised_alpha_prediction_network(Rnn):
    def __init__(self, input_shape, pad_value, rnn_units, dense_units,
                 rnn, dropout, implementation, bidirectional, batchnormalization,
                 gram, size_groups, lmbd):
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
        super().__init__(input_shape, pad_value, rnn_units, dense_units,
                         rnn, dropout, implementation,
                         bidirectional, batchnormalization)
        self.model = self.__create_RNN_columnar_network(gram, size_groups)

        self.hyperparams = {'lambda_start': lmbd / 10.,
                            'lambda_end': lmbd,
                            'end_epoch': 15}

        self.size_groups = size_groups

    def __create_RNN_columnar_network(self, gram, size_groups):
        """

        :return: Keras Deep RNN Siamese network
        :rtype: keras.models.Model
        """
        self.sparse_rate_callback = LambdaRateScheduler(start=self.hyperparams['lambda_start'],
                                                        end=self.hyperparams['lambda_end'],
                                                        end_epoch=self.hyperparams['end_epoch'])
        
        base_network = self.create_RNN_base_network()
        input_ = Input(shape=self.input_shape)
        processed = base_network(input_)
        parent = Dense(units=1, use_bias=False
                       if self.batchnormalization else True)(processed)
        if self.batchnormalization:
            parent = BatchNormalization()(parent)
        out = GroupSoftThresholdingLayer(self.size_groups)(parent)

        model = Model(input_, out)

        optimizer = Adam(clipnorm=1.)
        if self.gpu_count > 1:
            model = multi_gpu.make_parallel(model, self.gpu_count)

        loss_function = KSS_Loss(gram, size_groups, self.sparse_rate_callback.var)

        model.compile(loss=loss_function, optimizer=optimizer)

        return model

    def train_and_validate(self, trval_indices,
                           seqs,
                           alphas,
                           epochs,
                           patience,
                           logfile_hdf5):
        """Keras Siamese RNN training function.
        Carries out training and validation for given data over given number of epochs
        Logs results and network parameters

        :param trval_indices: Training and Validation 2-tuples of time series index pairs
        :param seqs: List of time series
        :param alphas: List of alpha for every seq
        :param epochs: Number of passes over data set
        :param patience: Early Stopping parameter
        :param logfile_hdf5: Log file name for network structure and weights in HDF5 format
        :type tr_indices: list of tuples
        :type seqs: list of np.ndarrays
        :type alphas: array of float
        :type epochs: int
        :type patience: int
        :type logfile_hdf5: str
        """
        trval_x = seqs[trval_indices]
        trval_y = alphas[trval_indices]

        tr_start = time.time()

        batch_size = 1024 * self.gpu_count

        mcp = ModelCheckpoint(filepath=logfile_hdf5, verbose=1, save_best_only=True)
        er = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience,
                           verbose=0, mode='auto')
        callbacks = [mcp, er, self.sparse_rate_callback]

        self.model.fit(trval_x, trval_y, batch_size=batch_size,
                       verbose=2, epochs=epochs,
                       validation_split=0.1, shuffle=True,
                       callbacks=callbacks)

        tr_end = time.time()
        return tr_start, tr_end

    def predict(self, te_indices, seqs):
        """Keras Siamese RNN prediction function.
        Carries out predicting for given data
        Logs results and network parameters

        :param te_indices: Testing 2-tuples of time series index pairs
        :param seqs: List of time series
        :return: Predictions

        :type te_indices: list of tuples
        :type seqs: list of np.ndarrays
        :returns: List of predicted network outputs
        :rtype: np.ndarrays
        """
        te_x = seqs[te_indices]

        # prediction
        pred_start = time.time()
        pred = self.model.predict(te_x)
        pred_end = time.time()

        return pred, pred_start, pred_end

class SoftThresholdingLayer(Layer):
    """Parametric Rectified Linear Unit.
    It follows:
    `f(x) = alpha * x for x < 0`,
    `f(x) = x for x >= 0`,
    where `alpha` is a learned array with the same shape as x.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        alpha_initializer: initializer function for the weights.
        alpha_regularizer: regularizer for the weights.
        alpha_constraint: constraint for the weights.
        shared_axes: the axes along which to share learnable
            parameters for the activation function.
            For example, if the incoming feature maps
            are from a 2D convolution
            with output shape `(batch, height, width, channels)`,
            and you wish to share parameters across space
            so that each filter only has one set of parameters,
            set `shared_axes=[1, 2]`.
    # References
        - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)
    """

    @interfaces.legacy_prelu_support
    def __init__(self, theta_initializer='zeros',
                 theta_regularizer=None,
                 theta_constraint=None,
                 shared_axes=None,
                 **kwargs):
        super(SoftThresholdingLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self.theta_initializer = initializers.get(theta_initializer)
        self.theta_regularizer = regularizers.get(theta_regularizer)
        self.theta_constraint = constraints.get(theta_constraint)
        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        self.param_broadcast = [False] * len(param_shape)
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
                self.param_broadcast[i - 1] = True
        self.theta = self.add_weight(shape=param_shape,
                                     name='theta',
                                     initializer=self.theta_initializer,
                                     regularizer=self.theta_regularizer,
                                     constraint=self.theta_constraint)
        # Set input spec
        axes = {}
        if self.shared_axes:
            for i in range(1, len(input_shape)):
                if i not in self.shared_axes:
                    axes[i] = input_shape[i]
        self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs, mask=None):
        return K.sign(inputs) * K.relu(K.abs(inputs) - self.theta)

    def get_config(self):
        config = {
            'theta_initializer': initializers.serialize(self.theta_initializer),
            'theta_regularizer': regularizers.serialize(self.theta_regularizer),
            'theta_constraint': constraints.serialize(self.theta_constraint),
            'shared_axes': self.shared_axes
        }
        base_config = super(SoftThresholdingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class GroupSoftThresholdingLayer(Layer):
    """Parametric Rectified Linear Unit.
    It follows:
    `f(x) = alpha * x for x < 0`,
    `f(x) = x for x >= 0`,
    where `alpha` is a learned array with the same shape as x.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        alpha_initializer: initializer function for the weights.
        alpha_regularizer: regularizer for the weights.
        alpha_constraint: constraint for the weights.
        shared_axes: the axes along which to share learnable
            parameters for the activation function.
            For example, if the incoming feature maps
            are from a 2D convolution
            with output shape `(batch, height, width, channels)`,
            and you wish to share parameters across space
            so that each filter only has one set of parameters,
            set `shared_axes=[1, 2]`.
    # References
        - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)
    """

    @interfaces.legacy_prelu_support
    def __init__(self,
                 size_groups,
                 theta_initializer='zeros',
                 theta_regularizer=None,
                 theta_constraint=None,
                 **kwargs):
        super(GroupSoftThresholdingLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self.theta_initializer = initializers.get(theta_initializer)
        self.theta_regularizer = regularizers.get(theta_regularizer)
        self.theta_constraint = constraints.get(theta_constraint)

        self.size_groups = size_groups
        self.cumsum = np.cumsum(size_groups)

    def build(self, input_shape):
        param_shape = [len(self.size_groups)]
        self.theta = self.add_weight(shape=param_shape,
                                     name='theta',
                                     initializer=self.theta_initializer,
                                     regularizer=self.theta_regularizer,
                                     constraint=self.theta_constraint)
        # Set input spec
        axes = {}
        self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs, mask=None):
        inputs_permute = K.permute_dimensions(inputs, [len(inputs.shape) - 1] + list(range(len(inputs.shape) - 1)))
        input_g = [inputs_permute[s:e] for (s, e) in zip(np.concatenate([np.array([0]), self.cumsum[:-1]]), self.cumsum)]
        input_g_norm = [K.sqrt(K.sum(K.square(g), keepdims=True) + K.epsilon()) for g in input_g]
        input_g_thres = [g / nrm * K.relu(nrm - t) for (g, nrm, t)
                         in zip(input_g, input_g_norm, self.theta)]
        outputs = K.permute_dimensions(input_g_thres, list(range(1, len(inputs.shape))) + [0])
        return outputs

    def get_config(self):
        config = {
            'theta_initializer': initializers.serialize(self.theta_initializer),
            'theta_regularizer': regularizers.serialize(self.theta_regularizer),
            'theta_constraint': constraints.serialize(self.theta_constraint),
        }
        base_config = super(GroupSoftThresholdingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class LambdaRateScheduler(Callback):
    '''Sparse rate scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            learning rate as output (float).
    '''
    def __init__(self, start, end, end_epoch):
        super(LambdaRateScheduler, self).__init__()
        self.var = K.variable(start, dtype=K.floatx(), name='k')
        self.start = start
        self.end = end
        self.end_epoch = end_epoch

    def on_epoch_begin(self, epoch, logs={}):
        l = np.min([epoch / self.end_epoch, 1.])
        lmbd = (1 - l) * self.start + l * self.end
        K.set_value(self.var, lmbd)
        print(K.get_value(self.var))

class KSS_Loss:
    def __init__(self, gram, size_groups, lmbd):
        self.gram = gram
        self.size_groups = size_groups
        self.lmbd = lmbd
    def __call__(self, k_true, alpha_pred):
        cumsum = np.cumsum(self.size_groups)

        quad = K.batch_dot(K.dot(alpha_pred.T, self.gram),
                           alpha_pred,
                           axis=-1)
        linear = K.batch_dot(k_true, alpha_pred, axis=-1)

        reg = K.stack([K.sum([K.sum(K.square(a[s:e]))
                              for (s, e)
                              in zip(np.concatenate([np.array([0]),
                                                     cumsum[:-1]]),
                              cumsum)])
                       for a in tf.unstack(alpha_pred)])
        return .5 * quad - linear + self.lmbd * reg



