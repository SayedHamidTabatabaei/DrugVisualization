from tensorflow.keras.layers import Dense, Input
# noinspection PyUnresolvedReferences
from tensorflow.keras.models import Sequential

from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from core.models.training_parameter_models.split_drugs_test_with_train_training_parameter_model import SplitDrugsTestWithTrainTrainingParameterModel
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.Test


class TrainPlanTest(TrainBaseService):
    def train(self, parameters: SplitDrugsTestWithTrainTrainingParameterModel) -> TrainingSummaryDTO:

        for x_train, x_test, y_train, y_test in super().k_fold_train_test_data(parameters.drug_data, parameters.interaction_data, padding=True):

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
