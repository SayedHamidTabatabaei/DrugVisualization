import tensorflow as tf
from tensorflow.keras import layers, Model

from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from core.models.training_parameter_models.split_interaction_similarities_training_parameter_model import SplitInteractionSimilaritiesTrainingParameterModel
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.Deep_DDI


class DeepDDI(Model):
    def __init__(self, hidden_size=2048, hidden_layers_num=9, output_size=65, dropout_prob=0.2):
        super(DeepDDI, self).__init__()

        assert hidden_layers_num > 1

        self.input_layer = layers.Dense(hidden_size, activation='relu')
        self.hidden_layers = [
            layers.Dense(hidden_size, activation='relu') for _ in range(hidden_layers_num - 1)
        ]
        self.dropout_layers = [
            layers.Dropout(dropout_prob) for _ in range(hidden_layers_num - 1)
        ]
        self.output_layer = layers.Dense(output_size, activation='sigmoid')

    def call(self, inputs):
        x = tf.concat(inputs, axis=1)
        x = self.input_layer(x)
        for hidden, dropout in zip(self.hidden_layers, self.dropout_layers):
            x = hidden(x)
            x = dropout(x)
        return self.output_layer(x)


class DeepDDITrainService(TrainBaseService):

    def __init__(self, category):
        super().__init__(category)
        self.drug_channels: int = 256
        self.hidden_channels: int = 2048
        self.hidden_layers_num: int = 9
        self.dropout_prob: float = 0.2

    def train(self, parameters: SplitInteractionSimilaritiesTrainingParameterModel) -> TrainingSummaryDTO:

        x_train, x_test, y_train, y_test = super().split_train_test(parameters.drug_data, parameters.interaction_data, padding=True, pca_generating=True)

        model = DeepDDI(hidden_size=self.hidden_channels,
                        hidden_layers_num=self.hidden_layers_num,
                        output_size=self.num_classes,
                        dropout_prob=self.dropout_prob)

        model.compile(optimizer="adam",
                      loss='binary_crossentropy',  # Multi-label classification
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train, epochs=50, batch_size=256, validation_data=(x_test, y_test))

        result = self.calculate_evaluation_metrics(model, x_test, y_test)

        self.plot_accuracy(history, parameters.train_id)
        self.plot_loss(history, parameters.train_id)

        result.model_info = self.get_model_info(model)

        if parameters.interaction_data is not None:
            result.data_report = self.get_data_report_split(parameters.interaction_data, y_train, y_test)

        return result
