import tensorflow as tf
from tensorflow.keras import layers, models
# noinspection PyUnresolvedReferences
from tensorflow.keras.models import Model

from businesses.trains.models.train_base_model import TrainBaseModel
from core.models.training_params import TrainingParams
from core.repository_models.training_drug_interaction_dto import TrainingDrugInteractionDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO


class SumSoftmaxesModel(TrainBaseModel):

    def __init__(self, train_id: int, num_classes: int, interaction_data: list[TrainingDrugInteractionDTO], training_params: TrainingParams):
        super().__init__(train_id, num_classes)
        self.interaction_data = interaction_data
        self.training_params = training_params

    def create_model(self, input_shape):
        model = models.Sequential()
        model.add(layers.Input(shape=input_shape))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(self.num_classes, activation='softmax'))  # Softmax output
        return model

    def build_model(self, x_train):

        models_list = [self.create_model(d.shape[1:]) for d in x_train]

        inputs = [tf.keras.Input(shape=d.shape[1:]) for d in x_train]

        softmax_outputs = [model(input_layer) for model, input_layer in zip(models_list, inputs)]

        summed_output = tf.keras.layers.Add()(softmax_outputs)

        model = tf.keras.Model(inputs=inputs, outputs=summed_output)

        return model

    def fit_model(self, x_train, y_train, x_val, y_val, x_test, y_test) -> TrainingSummaryDTO:

        model = self.build_model(x_train)

        return super().base_fit_model(model, self.training_params, self.interaction_data,
                                      x_train, y_train, x_val, y_val, x_test, y_test)
