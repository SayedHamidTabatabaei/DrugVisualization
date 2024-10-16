from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Concatenate, Flatten
from tensorflow.keras.models import Model
from tqdm import tqdm

from businesses.trains.train_base_service import TrainBaseService
from common.enums.category import Category
from common.enums.train_models import TrainModel
from core.models.data_params import DataParams
from core.models.training_parameter_models.split_interaction_similarities_training_parameter_model import SplitInteractionSimilaritiesTrainingParameterModel
from core.models.training_params import TrainingParams
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.MHA


class MhaTrainService(TrainBaseService):

    def __init__(self, category: TrainModel):
        super().__init__(category)

        self.text_embed_size = 512
        self.num_heads = 8
        self.encoding_dim = 128
        self.dnn_hidden_size = 128
        self.num_classes = 65

    def create_autoencoder(self, input_shape):
        input_layer = Input(shape=input_shape)
        encoded = Dense(self.encoding_dim, activation='relu')(input_layer)
        return input_layer, encoded

    def train(self, parameters: SplitInteractionSimilaritiesTrainingParameterModel) -> TrainingSummaryDTO:

        x_train, x_test, y_train, y_test = super().split_train_test(parameters.data)

        x_train, x_test = super().create_input_tensors_pad(x_train, x_test)

        input_layers = []
        encoded_models = []

        for idx, d in tqdm(enumerate(x_train), "Creating Encoded Models..."):

            shapes = {tuple(s.shape) for s in d}

            if len(shapes) != 1:
                raise ValueError(f"Error: Multiple shapes found: {shapes}")

            if parameters.data[idx][0].category in (Category.Substructure, Category.Pathway, Category.Target, Category.Enzyme):
                input_layer, encoded_model = self.create_autoencoder(input_shape=shapes.pop())
                input_layers.append(input_layer)
                encoded_models.append(encoded_model)
            else:
                input_layer = Input(shape=shapes.pop())
                attention_output = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.text_embed_size)(input_layer, input_layer)
                attention_output = LayerNormalization(epsilon=1e-6)(attention_output + input_layer)
                attention_output = Flatten()(attention_output)  # Flatten to (batch_size, -1)
                encoded_model = Dense(self.encoding_dim, activation='relu')(attention_output)
                # encoded_model = Dense(self.encoding_dim, activation='relu')(input_layer)
                input_layers.append(input_layer)
                encoded_models.append(encoded_model)

        concatenated = Concatenate()([encoded for encoded in encoded_models])

        # Define DNN layers after concatenation
        x = Dense(256, activation='relu')(concatenated)
        x = Dense(128, activation='relu')(x)
        output = Dense(self.num_classes, activation='softmax')(x)  # Softmax for multi-class classification

        # Create the full model
        full_model = Model(inputs=[input_layer for input_layer in input_layers], outputs=output)

        return super().fit_dnn_model(data_params=DataParams(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test),
                                     training_params=TrainingParams(train_id=parameters.train_id, optimizer='adam', loss=parameters.loss_function,
                                                                    class_weight=parameters.class_weight),
                                     model=full_model,
                                     data=parameters.data)
