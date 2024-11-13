from tensorflow.keras import layers, models

from businesses.trains.models.sum_softmaxes_model import SumSoftmaxesModel
from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from core.models.training_parameter_models.split_interaction_similarities_training_parameter_model import SplitInteractionSimilaritiesTrainingParameterModel
from core.models.training_params import TrainingParams
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.SumSoftmaxOutputs


class SumSoftmaxOutputsTrainService(TrainBaseService):

    def create_model(self, input_shape):
        model = models.Sequential()
        model.add(layers.Input(shape=input_shape))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(self.num_classes, activation='softmax'))  # Softmax output
        return model

    def train(self, parameters: SplitInteractionSimilaritiesTrainingParameterModel) -> TrainingSummaryDTO:

        x_train, x_test, y_train, y_test = super().split_train_test(parameters.drug_data, parameters.interaction_data, train_id=parameters.train_id,
                                                                    padding=True)

        training_params = TrainingParams(train_id=parameters.train_id, optimizer='adam', loss=parameters.loss_function, class_weight=parameters.class_weight)

        model = SumSoftmaxesModel(parameters.train_id, self.num_classes, parameters.interaction_data, training_params=training_params)
        result = model.fit_model(x_train, y_train, x_test, y_test, x_test, y_test)

        return result
