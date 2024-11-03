import tensorflow as tf
from tensorflow.keras import layers, Model


class DeepDDIModel(Model):
    def __init__(self, hidden_size=2048, hidden_layers_num=9, output_size=65, dropout_prob=0.2):
        super(DeepDDIModel, self).__init__()

        assert hidden_layers_num > 1

        self.model_layers = []
        for _ in range(hidden_layers_num):
            self.model_layers.extend([
                layers.Dense(hidden_size),
                layers.ReLU(),
                layers.LayerNormalization(),
                layers.ReLU(),
                layers.Dropout(rate=dropout_prob)
            ])

        self.output_layer = layers.Dense(output_size, activation='sigmoid')

    def call(self, inputs):
        x = tf.concat(inputs, axis=1)
        for model_layer in self.model_layers:
            x = model_layer(x)

        return self.output_layer(x)


