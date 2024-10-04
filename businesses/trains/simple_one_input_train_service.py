import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from core.models.data_params import DataParams
from core.models.training_params import TrainingParams
from core.models.training_parameter_model import TrainingParameterModel
from core.repository_models.training_data_dto import TrainingDataDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.SimpleOneInput


class SimpleOneInputTrainService(TrainBaseService):

    def train(self, parameters: TrainingParameterModel, data: list[list[TrainingDataDTO]]) -> TrainingSummaryDTO:

        data = data[0]

        # Prepare input for the model
        x_pairs = np.array([item.concat_values for item in data])

        # Example labels (replace this with your actual interaction data)
        y = np.array([item.interaction_type for item in data])

        x_train = []
        x_test = []
        y_train = []
        y_test = []

        stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in stratified_split.split(x_pairs, y):
            x_train, x_test = x_pairs[train_index], x_pairs[test_index]
            y_train, y_test = y[train_index], y[test_index]

        # Define the model
        model = Sequential()

        # Input layer
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))

        # Hidden layers
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(32, activation='relu'))

        # Output layer with softmax
        model.add(Dense(65, activation='softmax'))

        return super().fit_dnn_model(data_params=DataParams(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test),
                                     training_params=TrainingParams(train_id=parameters.train_id, optimizer='adam', loss=parameters.loss_function,
                                                                    class_weight=parameters.class_weight),
                                     model=model,
                                     data=data)

