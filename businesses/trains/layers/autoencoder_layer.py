from tensorflow.keras.layers import Dense, Layer


class AutoEncoderLayer(Layer):
    def __init__(self, encoding_dim, input_dim, **kwargs):
        super(AutoEncoderLayer, self).__init__(**kwargs)
        self.encoding_dim = encoding_dim
        self.input_dim = input_dim
        self.encoder = None
        self.decoder = None

    def build(self, input_shape):
        self.encoder = Dense(self.encoding_dim, activation='relu')
        self.decoder = Dense(self.input_dim, activation='sigmoid', name="reconstruction_output")

        super(AutoEncoderLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)

        return decoded, encoded

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.input_dim), (input_shape[0], self.encoding_dim)
