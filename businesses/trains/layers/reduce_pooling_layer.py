import tensorflow as tf
from tensorflow.keras.layers import Layer


class ReducePoolingLayer(Layer):
    def __init__(self, axis=1, pooling_mode="mean", **kwargs):
        """
        This layer aggregates node features into a graph-level feature by performing
        mean, max, or sum pooling across the specified axis (typically axis=1 for nodes).

        Args:
            axis: The axis along which to compute the pooling (usually axis=1 for node features).
            pooling_mode: The type of pooling to perform ("mean", "max", "sum").
        """
        super(ReducePoolingLayer, self).__init__(**kwargs)
        self.axis = axis
        self.pooling_mode = pooling_mode.lower()

        # Ensure the pooling mode is valid
        assert self.pooling_mode in ["mean", "max", "sum"], "Invalid pooling mode. Choose 'mean', 'max', or 'sum'."

    def call(self, inputs, **kwargs):
        if self.pooling_mode == "mean":
            return tf.reduce_mean(inputs, axis=self.axis)
        elif self.pooling_mode == "max":
            return tf.reduce_max(inputs, axis=self.axis)
        elif self.pooling_mode == "sum":
            return tf.reduce_sum(inputs, axis=self.axis)


# https://chatgpt.com/c/67499d86-a9c0-800b-915f-8f71aae1d6ad