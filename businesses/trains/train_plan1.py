import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

from businesses.trains.train_plan_base import TrainPlanBase
from common.enums.train_models import TrainModel
from core.models.training_parameter_model import TrainingParameterModel
from core.repository_models.training_data_dto import TrainingDataDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.SimpleOneInput


class TrainPlan1(TrainPlanBase):

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

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

        super().plot_accuracy(history, parameters.train_id)
        super().plot_loss(history, parameters.train_id)

        return super().calculate_evaluation_metrics(model, x_test, y_test)
