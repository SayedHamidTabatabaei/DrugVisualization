class HyperParams:
    def __init__(self, encoding_dim, gat_units, num_heads, dense_units, droprate, pooling_mode = None, batch_size=128,
                 lr_rate=1e-4, adam_beta=None, alpha=0.0, schedule_number=1):
        self.encoding_dim = encoding_dim
        self.gat_units = gat_units
        self.num_heads = num_heads
        self.dense_units = dense_units
        self.droprate = droprate
        self.pooling_mode = pooling_mode

        self.batch_size = batch_size
        self.lr_rate = lr_rate
        self.adam_beta = adam_beta
        self.alpha = alpha
        self.schedule_number = schedule_number