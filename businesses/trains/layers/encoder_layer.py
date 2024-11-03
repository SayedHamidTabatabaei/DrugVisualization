from tensorflow.keras.layers import Dense, Layer


class EncoderLayer(Layer):
    def __init__(self, encoding_dim, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.encoding_dim = encoding_dim
        self.encoder = None  # We'll define this in build()

    def build(self, input_shape):
        # Define the dense layer with an input dimension inferred from input_shape
        self.encoder = Dense(self.encoding_dim, activation='relu')
        super(EncoderLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # Ensure the inputs match what the encoder expects
        encoded = self.encoder(inputs)
        return encoded

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.encoding_dim
