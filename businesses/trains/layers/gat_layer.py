# import tensorflow as tf
# from tensorflow.keras.layers import Dense, Layer
#
#
# class GATLayer(Layer):
#     def __init__(self, units, num_heads, **kwargs):
#         super(GATLayer, self).__init__(**kwargs)
#         self.units = units
#         self.num_heads = num_heads
#         self.attention_heads = [
#             Dense(units, activation="relu") for _ in range(num_heads)
#         ]
#         self.attention_weights = [
#             Dense(1) for _ in range(num_heads)
#         ]
#
#     def call(self, inputs, **kwargs):
#         # inputs: (batch_size, num_nodes, feature_dim)
#
#         feature_matrix, adjacency_matrix = inputs
#
#         all_heads = []
#         for head, weight in zip(self.attention_heads, self.attention_weights):
#             projection = head(feature_matrix)  # (batch_size, num_nodes, units)
#
#             attention_logits = weight(projection)  # (batch_size, num_nodes, 1)
#             attention_logits = tf.matmul(attention_logits, tf.transpose(attention_logits, perm=[0, 2, 1]))
#
#             attention_logits = tf.where(adjacency_matrix > 0, attention_logits, -1e9)
#
#             attention_scores = tf.nn.softmax(attention_logits, axis=-1)  # (batch_size, num_nodes, num_nodes)
#
#             weighted_projection = tf.matmul(attention_scores, projection)  # (batch_size, num_nodes, units)
#             all_heads.append(weighted_projection)
#
#         # Concatenate attention heads or take their average
#         return tf.reduce_mean(tf.stack(all_heads, axis=0), axis=0)


import tensorflow as tf
from tensorflow.keras.layers import Layer


class GATLayer(Layer):
    def __init__(self, units, num_heads, dropout_rate=0.2, alpha=0.2, **kwargs):
        super(GATLayer, self).__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.alpha = alpha  # LeakyReLU angle

    def build(self, input_shape):
        feat_shape = input_shape[0][-1]
        # Create weight matrices for each attention head
        self.W = [self.add_weight(shape=(feat_shape, self.units),
                                  initializer='glorot_uniform',
                                  name=f'W_{i}')
                  for i in range(self.num_heads)]
        # Attention vectors for each head
        self.a = [self.add_weight(shape=(2 * self.units, 1),
                                  initializer='glorot_uniform',
                                  name=f'a_{i}')
                  for i in range(self.num_heads)]
        super(GATLayer, self).build(input_shape)

    def call(self, inputs, training=None):
        features, adjacency = inputs

        outputs = []
        for head in range(self.num_heads):
            # Linear transformation of input features
            Wh = tf.matmul(features, self.W[head])  # (N, units)

            # Prepare pairs of nodes for attention coefficients
            # Wh_repeated and Wh_repeated_t contain pairs of nodes
            Wh_repeated = tf.expand_dims(Wh, axis=1)  # (N, 1, units)
            Wh_repeated_t = tf.expand_dims(Wh, axis=0)  # (1, N, units)

            # Concatenate pairs of nodes
            pairs = tf.concat([
                tf.tile(Wh_repeated, [1, tf.shape(features)[0], 1]),
                tf.tile(Wh_repeated_t, [tf.shape(features)[0], 1, 1])
            ], axis=-1)  # (N, N, 2*units)

            # Compute attention coefficients
            e = tf.nn.leaky_relu(tf.squeeze(tf.matmul(pairs, self.a[head])), alpha=self.alpha)  # (N, N)

            # Mask attention coefficients using adjacency matrix
            mask = tf.where(adjacency > 0, e, -10e9)
            attention = tf.nn.softmax(mask, axis=-1)

            if training and self.dropout_rate > 0:
                attention = tf.nn.dropout(attention, self.dropout_rate)

            # Apply attention to features
            output = tf.matmul(attention, Wh)  # (N, units)
            outputs.append(output)

        # Combine outputs from all heads (average or concatenate)
        final_output = tf.reduce_mean(tf.stack(outputs), axis=0)
        return final_output