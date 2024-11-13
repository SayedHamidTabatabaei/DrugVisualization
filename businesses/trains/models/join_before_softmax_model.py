from tensorflow.keras import layers, models
# noinspection PyUnresolvedReferences
from tensorflow.keras.models import Model
from tqdm import tqdm

from businesses.trains.models.train_base_model import TrainBaseModel
from core.models.training_params import TrainingParams
from core.repository_models.training_drug_interaction_dto import TrainingDrugInteractionDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO


class JoinBeforeSoftmaxModel(TrainBaseModel):

    def __init__(self, train_id: int, num_classes: int, interaction_data: list[TrainingDrugInteractionDTO], training_params: TrainingParams):
        super().__init__(train_id, num_classes)
        self.interaction_data = interaction_data
        self.training_params = training_params

    def build_model(self, x_train):

        input_models = []
        input_layers = []

        for d in tqdm(x_train, "Creating Models..."):
            shapes = {tuple(s.shape) for s in d}

            assert len(shapes) == 1, ValueError(f"Error: Multiple shapes found: {shapes}")

            input_data = layers.Input(shape=shapes.pop())
            x_data = layers.Dense(64, activation='relu')(input_data)
            x_data = layers.Dense(32, activation='relu')(x_data)
            model = models.Model(inputs=input_data, outputs=x_data)

            input_models.append(model)
            input_layers.append(model.input)

        concatenated = layers.Concatenate()([model.output for model in input_models])

        x = layers.Dense(32, activation='relu')(concatenated)

        output = layers.Dense(self.num_classes, activation='softmax')(x)

        model = models.Model(inputs=input_layers, outputs=output)

        return model

    def fit_model(self, x_train, y_train, x_val, y_val, x_test, y_test) -> TrainingSummaryDTO:

        model = self.build_model(x_train)

        return super().base_fit_model(model, self.training_params, self.interaction_data,
                                      x_train, y_train, x_val, y_val, x_test, y_test)
