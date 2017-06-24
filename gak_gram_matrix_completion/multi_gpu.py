# Modified version of https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py

from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import Model

import tensorflow as tf

def make_parallel(model, gpu_count):
    """Data parallelize model on multiple GPUs.
    Each GPU is assigned a copy of the model
    Each GPU is processes its own subbatch (batch is divided evenly, remainder assigned to last GPU)
    Output subbatches are merged on CPU

    :param model: Keras model to parallelize
    :param gpu_count: Number of GPUs
    :type model: keras.engine.training.Model
    :type gpu_count: int
    :returns: Parallelized Keras model
    :rtype: keras.engine.training.Model
    """
    
    def get_slice(data, idx, parts):
        """Get subbatch of batch for a GPU.
        
        :param data: Batch to slice
        :param idx: Index of GPU
        :param parts: Number of GPUs
        :type data: tf.Tensor
        :type idx: int
        :type parts: int
        :returns: Subbatch
        :rtype: tf.Tensor
        """

        shape = tf.shape(data)
        stride = tf.concat([ shape[:1] // parts, shape[1:]*0 ],axis=0)
        start = stride * idx
        if idx < gpu_count - 1:
            size = tf.concat([ shape[:1] // parts, shape[1:] ],axis=0)
        else:
            size = tf.concat([ shape[:1] - start[0] , shape[1:] ],axis=0)
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    #Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)                

                outputs = model(inputs)
                
                if not isinstance(outputs, list):
                    outputs = [outputs]
                
                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))
            
        return Model(input=model.inputs, output=merged)

