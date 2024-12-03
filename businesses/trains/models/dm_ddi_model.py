import tensorflow as tf
import numpy as np

from businesses.trains.models.train_base_model import TrainBaseModel
from core.repository_models.training_summary_dto import TrainingSummaryDTO


# --- AE Model (Autoencoder) ---
class AE(tf.keras.Model):
    def __init__(self, n_input, n_enc_1, n_enc_2, n_z):
        super(AE, self).__init__()
        self.enc_1 = tf.keras.layers.Dense(n_enc_1, activation='relu')
        self.enc_2 = tf.keras.layers.Dense(n_enc_2, activation='relu')
        self.z_layer = tf.keras.layers.Dense(n_z)

        self.dec_1 = tf.keras.layers.Dense(n_enc_2, activation='relu')
        self.dec_2 = tf.keras.layers.Dense(n_enc_1, activation='relu')
        self.x_bar_layer = tf.keras.layers.Dense(n_input)

    def call(self, x):
        enc_h1 = self.enc_1(x)
        enc_h2 = self.enc_2(enc_h1)
        z = self.z_layer(enc_h2)  # Encoded vector
        dec_h1 = self.dec_1(z)
        dec_h2 = self.dec_2(dec_h1)
        x_bar = self.x_bar_layer(dec_h2)
        return x_bar, enc_h1, enc_h2, z


# --- Attention Layer ---
class Attention(tf.keras.Model):
    def __init__(self, in_size, hidden_size=128):
        super(Attention, self).__init__()
        self.project = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation='tanh'),
            tf.keras.layers.Dense(1, use_bias=False)
        ])

    def call(self, z):
        w = self.project(z)  # (batch_size, seq_len, 1)
        beta = tf.nn.softmax(w, axis=1)  # (batch_size, seq_len, 1)
        output = tf.reduce_sum(beta * z, axis=1)  # (batch_size, features)
        return output, beta


# --- GNN Layer ---
class GNNLayer(tf.keras.layers.Layer):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = self.add_weight(shape=(in_features, out_features), initializer='glorot_uniform')

    def call(self, features, adj, active=True):
        support = tf.matmul(features, self.weight)  # (batch_size, out_features)
        output = tf.sparse.sparse_dense_matmul(adj, support)  # (batch_size, out_features)
        if active:
            output = tf.nn.relu(output)
        return output


# --- Combine Part Pairs ---
def combine_part_pairs(embedding, array, method_num):
    feature_pairs_train = []
    for pair in array:
        leftdrug = embedding[pair[0]]
        rightdrug = embedding[pair[1]]
        if method_num == 1:
            B = (leftdrug + rightdrug) / 2
        elif method_num == 2:
            B = leftdrug * rightdrug
        elif method_num == 3:
            B = leftdrug - rightdrug
        elif method_num == 4:
            B = tf.concat([leftdrug, rightdrug], axis=-1)
        feature_pairs_train.append(B)
    embedding_pairs = tf.stack(feature_pairs_train)  # Stack the list to form tensor (e.g., [37264, 65])
    return embedding_pairs


# --- Combine Drug Pairs ---
def combine_drugpairs(embedding_drug, method_num, adj_file):
    # Load adjacency matrix and labels (this function should be implemented separately)
    ddi_arr, label = load_adj(adj_file)  # Traverse adj matrix to get ddi_arr and label
    label = tf.convert_to_tensor(label, dtype=tf.int64)

    all_edge_array = ddi_arr
    num_train = int(np.floor(len(all_edge_array) * 0.8))

    # Split the edge array into training and test sets
    train_edge_array = all_edge_array[:num_train]  # First 80%
    test_edge_array = all_edge_array[num_train:]  # Remaining 20%

    # Split labels into training and test sets
    label_train_y = label[:num_train]
    label_test_y = label[num_train:]

    # Combine drug pairs using the specified method
    C1 = combine_part_pairs(embedding_drug, train_edge_array, method_num)
    C2 = combine_part_pairs(embedding_drug, test_edge_array, method_num)

    return C1, C2, label_train_y, label_test_y


# --- DM_DDI Model ---
class DM_DDI(tf.keras.Model):
    def __init__(self, n_input, n_enc_1, n_enc_2, n_z, pretrain_path=None, v=1):
        super(DM_DDI, self).__init__()
        self.ae = AE(n_input=1716, n_enc_1=2000, n_enc_2=256, n_z=n_z)

        # Optionally load pre-trained weights
        if pretrain_path:
            self.ae.load_weights(pretrain_path)

        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_z)

        self.attention = Attention(n_z)
        self.predict = tf.keras.layers.Dense(n_z)

        self.cluster_layer = self.add_weight(shape=(n_z, n_enc_2), initializer='glorot_normal')
        self.v = v

    def call(self, x, adj, method_num=1, adj_file="path_to_adj_file"):
        # AE Module
        x_bar, tra1, tra2, z = self.ae(x)
        sigma = 0.5
        h1 = self.gnn_1(x, adj)
        h2 = self.gnn_2((1 - sigma) * h1 + sigma * tra1, adj)
        h3 = self.gnn_3((1 - sigma) * h2 + sigma * tra2, adj)
        emb = tf.stack([h3, z], axis=1)  # (batch_size, 2, n_z)

        # Attention Fusion
        emb1, att1 = self.attention(emb)

        # Combine drug pairs
        C1, C2, label_train_y, label_test_y = combine_drugpairs(emb1, method_num, adj_file)

        return emb1, att1, x_bar, C1, C2, label_train_y, label_test_y


# --- Loss Function ---
def compute_loss(y_true, y_pred, x_true, x_pred):
    # Classification Loss (e.g., using cross-entropy)
    classification_loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_true, y_pred))

    # Reconstruction Loss (for the AE part)
    reconstruction_loss = tf.reduce_mean(tf.square(x_true - x_pred))

    # Combine the losses
    total_loss = classification_loss + reconstruction_loss
    return total_loss


# --- Load Adjacency Matrix ---
def load_adj(adj_file):
    # Example implementation: You can load the adjacency matrix from a file or compute it programmatically.
    # Here, `ddi_arr` is an array of edge pairs, and `label` contains corresponding labels.
    ddi_arr = np.array([[0, 1], [1, 2], [2, 3]])  # Example adjacency pairs
    label = np.array([0, 1, 1])  # Example labels
    return ddi_arr, label


class DMDDIModel(TrainBaseModel):

    def __init__(self, num_classes: int, train_id):
        super().__init__(train_id, num_classes)

    def fit_model(self, x_train, y_train, x_val, y_val, x_test, y_test) -> TrainingSummaryDTO:

        model = DM_DDI(n_input=1716, n_enc_1=2000, n_enc_2=256, n_z=64)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                      loss=compute_loss)

        # Fit the model
        history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_val, y_val))

        self.save_plots(history, self.train_id)

        y_pred = model.predict(x_test)

        result = self.calculate_evaluation_metrics(model, x_test, y_test, y_pred=y_pred)

        result.model_info = self.get_model_info(model)

        return result
