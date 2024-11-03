from tensorflow.keras import layers, Model


class DDIMDLModel(Model):
    def __init__(self, droprate=0.3, num_classes=65):

        super(DDIMDLModel, self).__init__()

        self.input_layer = layers.Dense(512, activation='relu')
        self.batch_norm1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(droprate)

        self.hidden_layer = layers.Dense(256, activation='relu')
        self.batch_norm2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(droprate)

        self.output_layer = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.batch_norm1(x)
        x = self.dropout1(x)

        x = self.hidden_layer(x)
        x = self.batch_norm2(x)
        x = self.dropout2(x)

        return self.output_layer(x)
