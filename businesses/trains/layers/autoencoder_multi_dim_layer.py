from tensorflow.keras.layers import Layer, Dense, Reshape, Flatten


class AutoEncoderMultiDimLayer(Layer):
    def __init__(self, encoding_dim, target_shape, **kwargs):
        super(AutoEncoderMultiDimLayer, self).__init__(**kwargs)
        self.encoding_dim = encoding_dim
        self.target_shape = target_shape
        self.encoder = None
        self.decoder = None

    def build(self, input_shape):

        self.flatten = Flatten()  # Flatten input data (768, 512) -> 1D vector
        self.encoder = Dense(self.encoding_dim[0] * self.encoding_dim[1], activation='relu')  # Latent space size = (196, 128)

        # Decoder: Reconstruct the input from the encoded representation
        self.decoder = Dense(input_shape[1] * input_shape[2], activation='sigmoid')  # Reconstruction to 768 * 512
        self.reshape = Reshape((input_shape[1], input_shape[2]))  # Reshape back to (768, 512)

        super(AutoEncoderMultiDimLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # Flatten the input before feeding into the encoder
        flattened_input = self.flatten(inputs)

        # Encode the input into a latent space of shape (196, 128)
        encoded = self.encoder(flattened_input)

        # Decode the latent space back into a flattened vector (768 * 512)
        decoded = self.decoder(encoded)

        # Reshape the decoded output to the target shape (768, 512) or (196, 128) as needed
        decoded = self.reshape(decoded)

        return decoded, encoded

    def compute_output_shape(self, input_shape):
        # Output shape will be (768, 512) (reconstructed) and (196, 128) (encoded)
        return (input_shape[0], self.target_shape[0], self.target_shape[1]), (input_shape[0], self.encoding_dim[0] * self.encoding_dim[1])
