class HyperParams:
    def __init__(self, encoding_dim, gat_units, num_heads, dense_units, droprate):
        self.encoding_dim = encoding_dim
        self.gat_units = gat_units
        self.num_heads = num_heads
        self.dense_units = dense_units
        self.droprate = droprate