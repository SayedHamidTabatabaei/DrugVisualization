import tensorflow as tf
from tensorflow.keras import layers, models
from tqdm import tqdm

from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from core.models.data_params import DataParams
from core.models.training_params import TrainingParams
from core.models.training_parameter_model import TrainingParameterModel
from core.repository_models.training_data_dto import TrainingDataDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.Contact_DNN


class ConcatDnnTrainService(TrainBaseService):

    def train(self, parameters: TrainingParameterModel, data: list[list[TrainingDataDTO]]) -> TrainingSummaryDTO:
        x_train, x_test, y_train, y_test = super().split_train_test(data)

        x_train = tf.concat([x for x in tqdm(x_train, "Concat train data")], axis=1)
        x_test = tf.concat([x for x in tqdm(x_test, "Concat test data")], axis=1)

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
                                     data=data)

