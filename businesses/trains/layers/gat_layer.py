import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer


class GATLayer(Layer):
    def __init__(self, units, num_heads, **kwargs):
        super(GATLayer, self).__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.attention_heads = [
            Dense(units, activation="relu") for _ in range(num_heads)
        ]
        self.attention_weights = [
            Dense(1) for _ in range(num_heads)
        ]

    def call(self, inputs, adjacency_matrix=None, **kwargs):
        # inputs: (batch_size, num_nodes, feature_dim)

        feature_matrix, adjacency_matrix = inputs

        all_heads = []
        for head, weight in zip(self.attention_heads, self.attention_weights):
            projection = head(feature_matrix)  # (batch_size, num_nodes, units)

            attention_logits = weight(projection)  # (batch_size, num_nodes, 1)
            attention_logits = tf.matmul(attention_logits, tf.transpose(attention_logits, perm=[0, 2, 1]))

            attention_logits = tf.where(adjacency_matrix > 0, attention_logits, -1e9)

            attention_scores = tf.nn.softmax(attention_logits, axis=-1)  # (batch_size, num_nodes, num_nodes)

            weighted_projection = tf.matmul(attention_scores, projection)  # (batch_size, num_nodes, units)
            all_heads.append(weighted_projection)

        # Concatenate attention heads or take their average
        return tf.reduce_mean(tf.stack(all_heads, axis=0), axis=0)
