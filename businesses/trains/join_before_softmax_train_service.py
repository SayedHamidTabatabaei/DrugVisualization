from tensorflow.keras import layers, models
from tqdm import tqdm

from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from core.models.data_params import DataParams
from core.models.training_parameter_models.split_interaction_similarities_training_parameter_model import SplitInteractionSimilaritiesTrainingParameterModel
from core.models.training_params import TrainingParams
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.JoinBeforeSoftmax


class JoinBeforeSoftmaxTrainService(TrainBaseService):

    @staticmethod
    def create_model(input_shape):
        input_data = layers.Input(shape=input_shape)
        x_data = layers.Dense(64, activation='relu')(input_data)
        x_data = layers.Dense(32, activation='relu')(x_data)
        return models.Model(inputs=input_data, outputs=x_data)

    def train(self, parameters: SplitInteractionSimilaritiesTrainingParameterModel) -> TrainingSummaryDTO:

        x_train, x_test, y_train, y_test = super().split_train_test(parameters.data)

        x_train, x_test = super().create_input_tensors_pad(x_train, x_test)

        input_models = []
        input_layers = []

        for d in tqdm(x_train, "Creating Models..."):
            shapes = {tuple(s.shape) for s in d}

            assert len(shapes) == 1, ValueError(f"Error: Multiple shapes found: {shapes}")

            model = self.create_model(shapes.pop())
            input_models.append(model)
            input_layers.append(model.input)

        # Concatenate both models' outputs
        concatenated = layers.Concatenate()([model.output for model in input_models])

        # Add a fully connected layer before softmax
        x = layers.Dense(32, activation='relu')(concatenated)

        # Output layer with softmax activation
        output = layers.Dense(self.num_classes, activation='softmax')(x)

        # Create the final model
        final_model = models.Model(inputs=input_layers, outputs=output)

        return super().fit_dnn_model(data_params=DataParams(x_train=[x for x in x_train], y_train=y_train, x_test=[x for x in x_test], y_test=y_test),
                                     training_params=TrainingParams(train_id=parameters.train_id, optimizer='adam', loss=parameters.loss_function,
                                                                    class_weight=parameters.class_weight),
                                     model=final_model,
                                     data=parameters.data)
