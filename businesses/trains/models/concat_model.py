from tensorflow.keras import layers, models
# noinspection PyUnresolvedReferences
from tensorflow.keras.models import Model

from businesses.trains.models.train_base_model import TrainBaseModel
from core.models.training_params import TrainingParams
from core.repository_models.training_drug_interaction_dto import TrainingDrugInteractionDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO


class ConcatModel(TrainBaseModel):

    def __init__(self, train_id: int, num_classes: int, interaction_data: list[TrainingDrugInteractionDTO], training_params: TrainingParams):
        super().__init__(train_id, num_classes)
        self.interaction_data = interaction_data
        self.training_params = training_params

    def build_model(self, input_shape):

        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=(input_shape,)))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        return model

    def fit_model(self, x_train, y_train, x_val, y_val, x_test, y_test) -> TrainingSummaryDTO:

        model = self.build_model(x_train.shape[1])

        return super().base_fit_model(model, self.training_params, self.interaction_data,
                                      x_train, y_train, x_val, y_val, x_test, y_test)
