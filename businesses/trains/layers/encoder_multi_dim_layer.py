from tensorflow.keras.layers import Layer, Dense, Reshape, Flatten


class EncoderMultiDimLayer(Layer):
    def __init__(self, encoding_dim, target_shape, **kwargs):
        super(EncoderMultiDimLayer, self).__init__(**kwargs)
        self.flatten = None
        self.encoder_reshape = None
        self.encoding_dim = encoding_dim
        self.target_shape = target_shape
        self.encoder = None
        self.decoder = None

    def build(self, input_shape):

        self.flatten = Flatten()  # Flatten input data (768, 512) -> 1D vector
        self.encoder = Dense(self.encoding_dim[0] * self.encoding_dim[1], activation='relu')  # Latent space size = (196, 128)
        self.encoder_reshape = Reshape((self.encoding_dim[0], self.encoding_dim[1]))  # Reshape back to (196, 128)

        super(EncoderMultiDimLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # Flatten the input before feeding into the encoder
        flattened_input = self.flatten(inputs)

        # Encode the input into a latent space of shape (196, 128)
        encoded = self.encoder(flattened_input)
        encoded = self.encoder_reshape(encoded)

        return encoded

    def compute_output_shape(self, input_shape):
        # Output shape will be (768, 512) (reconstructed) and (196, 128) (encoded)
        return input_shape[0], self.encoding_dim[0] * self.encoding_dim[1]
