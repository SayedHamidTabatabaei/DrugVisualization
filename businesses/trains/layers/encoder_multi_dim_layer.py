from tensorflow.keras.layers import Layer, Dense, Reshape, Flatten, BatchNormalization, Dropout


class EncoderMultiDimLayer(Layer):
    def __init__(self, encoding_dim, source_shape, **kwargs):
        super(EncoderMultiDimLayer, self).__init__(**kwargs)
        self.dropout = None
        self.batch_normalization = None
        self.batch_normalization_1 = None
        self.dropout_1 = None
        self.encoder_1 = None
        self.flatten = None
        self.encoder_reshape = None
        self.encoding_dim = encoding_dim
        self.source_shape = source_shape
        self.encoder = None
        self.decoder = None

    def build(self, input_shape):

        self.flatten = Flatten()  # Flatten input data (768, 512) -> 1D vector

        if self.encoding_dim == 2:
            self.encoder = Dense(self.encoding_dim[0] * self.encoding_dim[1], activation='relu')  # Latent space size = (196, 128)
            self.encoder_reshape = Reshape((self.encoding_dim[0], self.encoding_dim[1]))  # Reshape back to (196, 128)
        else:
            self.encoder_1 = Dense(input_shape[-1], activation='relu')
            self.batch_normalization_1 = BatchNormalization()
            self.dropout_1 = Dropout(0.5)

            self.encoder = Dense(self.encoding_dim, activation='relu')
            self.batch_normalization = BatchNormalization()
            self.dropout = Dropout(0.5)

        super(EncoderMultiDimLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # Flatten the input before feeding into the encoder
        flattened_input = self.flatten(inputs)

        # Encode the input into a latent space of shape (196, 128)
        if self.encoding_dim == 2:
            encoded = self.encoder(flattened_input)
            encoded = self.encoder_reshape(encoded)
        else:
            encoded = self.encoder_1(flattened_input)
            encoded = self.batch_normalization_1(encoded)
            encoded = self.dropout_1(encoded)

            encoded = self.encoder(encoded)
            encoded = self.batch_normalization(encoded)
            encoded = self.dropout(encoded)

        return encoded

    def compute_output_shape(self, input_shape):
        # Output shape will be (768, 512) (reconstructed) and (196, 128) (encoded)
        if self.encoding_dim == 2:
            return input_shape[0], self.encoding_dim[0] * self.encoding_dim[1]
        else:
            return input_shape[0], self.encoding_dim
