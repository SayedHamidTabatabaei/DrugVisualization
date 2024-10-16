from tensorflow.keras import layers, models

from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from core.models.data_params import DataParams
from core.models.training_parameter_models.split_interaction_similarities_training_parameter_model import SplitInteractionSimilaritiesTrainingParameterModel
from core.models.training_params import TrainingParams
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.Contact_DNN


class ConcatDnnTrainService(TrainBaseService):

    def train(self, parameters: SplitInteractionSimilaritiesTrainingParameterModel) -> TrainingSummaryDTO:
        x_train, x_test, y_train, y_test = super().split_train_test(parameters.data)

        x_train, x_test = super().create_input_tensors_flat(x_train, x_test)

        input_shape = x_train.shape[1]

        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=(input_shape,)))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        return super().fit_dnn_model(data_params=DataParams(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test),
                                     training_params=TrainingParams(train_id=parameters.train_id, optimizer='adam', loss=parameters.loss_function,
                                                                    class_weight=parameters.class_weight),
                                     model=model,
                                     data=parameters.data)
