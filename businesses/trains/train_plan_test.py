import numpy as np
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

from businesses.trains.train_plan_base import TrainPlanBase
from common.enums.train_models import TrainModel
from core.models.training_parameter_model import TrainingParameterModel
from core.repository_models.training_data_dto import TrainingDataDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.Test


class TrainPlanTest(TrainPlanBase):
    def train(self, parameters: TrainingParameterModel, data: list[list[TrainingDataDTO]]) -> TrainingSummaryDTO:
        y = [y.interaction_type for y in data[0]]
        x_concat_values = [[dto.concat_values for dto in set_data] for set_data in data]

        x_ragged_sets = [tf.ragged.constant(x) for x in x_concat_values]

        num_samples = len(x_ragged_sets[0])  # Assumes all sets have the same number of items

        # Stratified K-Fold
        k_folds = 5
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

        for train_index, test_index in skf.split(np.zeros(num_samples), y):
            x_train_sets = [x_ragged[train_index] for x_ragged in x_ragged_sets]
            x_test_sets = [x_ragged[test_index] for x_ragged in x_ragged_sets]

            y_train = y[train_index]
            y_test = y[test_index]

            max_length = max([x.shape[1] for x in x_train_sets])
            x_train_sets_padded = [x.to_tensor(shape=[None, max_length]) for x in x_train_sets]
            x_test_sets_padded = [x.to_tensor(shape=[None, max_length]) for x in x_test_sets]

            # Example DNN model
            input_shape = x_train_sets_padded[0].shape[1:]  # Shape from one padded set
            model = Sequential([
                Input(shape=input_shape),
                Dense(128, activation='relu'),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(65, activation='softmax')  # Output layer for 65 categories
            ])

            # Compile the model
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Train the model on the first set (for example) and evaluate
            model.fit(x_train_sets_padded[0], y_train, epochs=20, batch_size=32, validation_data=(x_test_sets_padded[0], y_test))
