from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import pandas as pd
import numpy as np

# from sklearn.decomposition import PCA
#
#
# def generate_pca(values: list[float]):
#     number_of_features = 31
#     array_np = np.array(values)
#     num_samples = len(array_np) // number_of_features
#     array_reshaped = array_np[:num_samples * number_of_features].reshape(num_samples, number_of_features)
#
#     pca = PCA(n_components=min(32, number_of_features))
#     reduced_array = pca.fit_transform(array_reshaped)
#
#     print("Reduced Array Shape:", reduced_array.shape)


def generate_autoencoder(values: list[float]):
    data_scaled = np.array(values)

    input_dim = data_scaled.shape[1]
    encoding_dim = 10

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Train the model
    autoencoder.fit(data_scaled, data_scaled, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

    encoder = Model(inputs=input_layer, outputs=encoded)
    encoded_data = encoder.predict(data_scaled)

    encoded_df = pd.DataFrame(encoded_data)
