import tensorflow as tf
from tensorflow.keras import layers, Model

from businesses.trains.models.train_base_model import TrainBaseModel
from core.repository_models.training_drug_interaction_dto import TrainingDrugInteractionDTO


class DeepDDIModel(Model):
    def __init__(self, hidden_size=2048, hidden_layers_num=9, output_size=65, dropout_prob=0.2):
        super(DeepDDIModel, self).__init__()

        assert hidden_layers_num > 1

        self.model_layers = []
        for _ in range(hidden_layers_num):
            self.model_layers.extend([
                layers.Dense(hidden_size),
                layers.ReLU(),
                layers.LayerNormalization(),
                layers.ReLU(),
                layers.Dropout(rate=dropout_prob)
            ])

        self.output_layer = layers.Dense(output_size, activation='sigmoid')

    def call(self, inputs, **kwargs):
        x = tf.concat(inputs, axis=1)
        for model_layer in self.model_layers:
            x = model_layer(x)

        return self.output_layer(x)

    def compile_model(self, optimizer='adam', loss='binary_crossentropy', metrics=None):
        if metrics is None:
            metrics = ['accuracy']

        self.compile(optimizer=optimizer, loss=loss, metrics=metrics)


class DeepDDITrainModel(TrainBaseModel):
    def __init__(self, train_id: int, num_classes: int, interaction_data: list[TrainingDrugInteractionDTO]):
        super().__init__(train_id, num_classes)
        self.interaction_data = interaction_data

        self.drug_channels: int = 256
        self.hidden_channels: int = 2048
        self.hidden_layers_num: int = 9
        self.dropout_prob: float = 0.2

    def fit_model(self, x_train, y_train, x_val, y_val, x_test, y_test):

        model = DeepDDIModel(hidden_size=self.hidden_channels,
                             hidden_layers_num=self.hidden_layers_num,
                             output_size=self.num_classes,
                             dropout_prob=self.dropout_prob)

        model.compile_model()

        history = model.fit(x_train, y_train, epochs=100, batch_size=256, validation_data=(x_val, y_val))

        result = self.calculate_evaluation_metrics(model, x_test, y_test)

        self.save_plots(history, self.train_id)

        result.model_info = self.get_model_info(model)

        if self.interaction_data is not None:
            result.data_report = self.get_data_report_split(self.interaction_data, y_train, y_test)

        return result
