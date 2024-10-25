import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Layer, Dropout
# noinspection PyUnresolvedReferences
from tensorflow.keras.models import Model

from businesses.trains.train_base_service import TrainBaseService
from common.enums.category import Category
from common.enums.train_models import TrainModel
from core.models.data_params import DataParams
from core.models.training_parameter_models.split_interaction_similarities_training_parameter_model import SplitInteractionSimilaritiesTrainingParameterModel
from core.models.training_params import TrainingParams
from core.repository_models.training_drug_data_dto import TrainingDrugDataDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.GAT_AE_Con_DNN


class GATLayer(Layer):
    def __init__(self, units, num_heads, **kwargs):
        super(GATLayer, self).__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.attention_heads = [
            Dense(units, activation="relu") for _ in range(num_heads)
        ]
        self.attention_weights = [
            Dense(1) for _ in range(num_heads)
        ]

    def call(self, inputs, **kwargs):
        # inputs: (batch_size, num_nodes, feature_dim)
        all_heads = []
        for head, weight in zip(self.attention_heads, self.attention_weights):
            projection = head(inputs)  # (batch_size, num_nodes, units)
            attention_logits = weight(projection)  # (batch_size, num_nodes, 1)
            attention_scores = tf.nn.softmax(attention_logits, axis=1)  # (batch_size, num_nodes, 1)
            weighted_projection = projection * attention_scores  # (batch_size, num_nodes, units)
            all_heads.append(weighted_projection)

        # Concatenate attention heads or take their average
        return tf.reduce_mean(tf.stack(all_heads, axis=0), axis=0)


class AutoencoderLayer(Layer):
    def __init__(self, encoding_dim, **kwargs):
        super(AutoencoderLayer, self).__init__(**kwargs)
        self.encoding_dim = encoding_dim
        self.encoder = None  # We'll define this in build()

    def build(self, input_shape):
        # Define the dense layer with an input dimension inferred from input_shape
        self.encoder = Dense(self.encoding_dim, activation='relu')
        super(AutoencoderLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # Ensure the inputs match what the encoder expects
        encoded = self.encoder(inputs)
        return encoded

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.encoding_dim


class ReduceMeanLayer(Layer):
    def __init__(self, axis=1, **kwargs):
        super(ReduceMeanLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=self.axis)


class GatAeConDnnTrainService(TrainBaseService):

    def __init__(self, category: TrainModel):
        super().__init__(category)
        self.encoding_dim = 128
        self.gat_units = 64
        self.num_heads = 4
        self.dense_units = [128, 64]

    def build_model(self, data: TrainingDrugDataDTO, x_train):

        encoded_models_1 = []
        encoded_models_2 = []
        input_layers_1 = []
        input_layers_2 = []
        smiles_input_1 = None
        smiles_input_2 = None
        gat_output_1 = None
        gat_output_2 = None

        gat_layer = GATLayer(units=self.gat_units, num_heads=self.num_heads)

        for idx, d in enumerate(data.train_values):
            if d.category == Category.Substructure:

                smiles_input_shape = x_train[idx][0].shape

                # SMILES input (graph-like input)
                smiles_input_1 = Input(shape=smiles_input_shape, name="Drug1_SMILES_Input")
                smiles_input_2 = Input(shape=smiles_input_shape, name="Drug2_SMILES_Input")

                # GAT processing for SMILES
                gat_output_1 = gat_layer(smiles_input_1)
                gat_output_1 = ReduceMeanLayer(axis=1)(gat_output_1)  # Aggregate over all nodes

                # GAT processing for SMILES
                gat_output_2 = gat_layer(smiles_input_2)
                gat_output_2 = ReduceMeanLayer(axis=1)(gat_output_2)  # Aggregate over all nodes

            else:
                autoencoder_layer = AutoencoderLayer(encoding_dim=self.encoding_dim)

                input_layer_1 = Input(shape=x_train[idx][0].shape)
                encoded_model_1 = autoencoder_layer(input_layer_1)
                input_layers_1.append(input_layer_1)
                encoded_models_1.append(encoded_model_1)

                input_layer_2 = Input(shape=x_train[idx][0].shape)
                encoded_model_2 = autoencoder_layer(input_layer_2)
                input_layers_2.append(input_layer_2)
                encoded_models_2.append(encoded_model_2)

        # Concatenate GAT outputs with their respective feature inputs
        combined_drug_1 = Concatenate()([gat_output_1] + encoded_models_1)
        combined_drug_2 = Concatenate()([gat_output_2] + encoded_models_2)

        # Combine both drugs' information for interaction prediction
        combined = Concatenate()([combined_drug_1, combined_drug_2])

        # Dense layers for final prediction
        x = combined
        for units in self.dense_units:
            x = Dense(units, activation="relu")(x)
            x = Dropout(0.3)(x)

        # Output layer (example for binary classification, use appropriate activation and units for your case)
        output = Dense(self.num_classes, activation="softmax")(x)

        model_inputs = [smiles_input_1] + input_layers_1 + [smiles_input_2] + input_layers_2

        model = Model(inputs=model_inputs, outputs=output)

        return model

    def train(self, parameters: SplitInteractionSimilaritiesTrainingParameterModel) -> (TrainingSummaryDTO, object):

        x_train, x_test, y_train, y_test = super().split_deepface_train_test(parameters.drug_data, parameters.interaction_data, padding=False)

        model = self.build_model(parameters.drug_data[0], x_train)

        return super().fit_dnn_model(data_params=DataParams(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test),
                                     training_params=TrainingParams(train_id=parameters.train_id, optimizer='adam', loss=parameters.loss_function,
                                                                    class_weight=parameters.class_weight),
                                     model=model,
                                     interactions=parameters.interaction_data)
