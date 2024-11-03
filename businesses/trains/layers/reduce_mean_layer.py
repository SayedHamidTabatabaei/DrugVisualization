import tensorflow as tf
from tensorflow.keras.layers import Layer


class ReduceMeanLayer(Layer):
    def __init__(self, axis=1, **kwargs):
        super(ReduceMeanLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=self.axis)