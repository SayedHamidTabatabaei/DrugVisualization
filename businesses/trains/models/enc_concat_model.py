from typing import Any

from tensorflow.keras.layers import Input, Dense, Concatenate
# noinspection PyUnresolvedReferences
from tensorflow.keras.models import Model
from tqdm import tqdm

from businesses.trains.models.train_base_model import TrainBaseModel
from core.models.training_params import TrainingParams
from core.repository_models.training_drug_interaction_dto import TrainingDrugInteractionDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO


class EncConcatModel(TrainBaseModel):

    def __init__(self, train_id: int, num_classes: int, interaction_data: list[TrainingDrugInteractionDTO], training_params: TrainingParams):
        super().__init__(train_id, num_classes)
        self.encoding_dim = 128
        self.interaction_data = interaction_data
        self.training_params = training_params

    def create_encoder(self, input_shape):
        input_layer = Input(input_shape)
        encoded = Dense(self.encoding_dim, activation='relu')(input_layer)
        return input_layer, encoded

    def build_model(self, x_train):
        input_layers: list[Any] = []
        encoded_models = []

        for d in tqdm(x_train, "Creating Encoded Models..."):

            shapes = {tuple(s.shape) for s in d}

            if len(shapes) != 1:
                raise ValueError(f"Error: Multiple shapes found: {shapes}")

            input_layer, encoded_model = self.create_encoder(shapes.pop())
            input_layers.append(input_layer)
            encoded_models.append(encoded_model)

        concatenated = Concatenate()([encoded for encoded in encoded_models])

        # Define DNN layers after concatenation
        x = Dense(256, activation='relu')(concatenated)
        x = Dense(128, activation='relu')(x)
        output = Dense(self.num_classes, activation='softmax')(x)  # Softmax for multi-class classification

        model = Model(inputs=[input_layer for input_layer in input_layers], outputs=output)

        return model

    def fit_model(self, x_train, y_train, x_val, y_val, x_test, y_test) -> TrainingSummaryDTO:

        model = self.build_model(x_train)

        return super().base_fit_model(model, self.training_params, self.interaction_data,
                                      x_train, y_train, x_val, y_val, x_test, y_test)
