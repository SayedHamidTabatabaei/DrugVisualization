from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization, Activation
# noinspection PyUnresolvedReferences
from tensorflow.keras.models import Model

from businesses.trains.layers.autoencoder_layer import AutoEncoderLayer
from businesses.trains.layers.gat_layer import GATLayer
from businesses.trains.layers.reduce_mean_layer import ReduceMeanLayer
from businesses.trains.train_base_service import TrainBaseService
from common.enums.category import Category
from common.enums.train_models import TrainModel
from core.models.data_params import DataParams
from core.models.training_parameter_models.split_interaction_similarities_training_parameter_model import SplitInteractionSimilaritiesTrainingParameterModel
from core.models.training_params import TrainingParams
from core.repository_models.training_drug_data_dto import TrainingDrugDataDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.GAT_AE_DNN


class GatAeDnnTrainService(TrainBaseService):

    def __init__(self, category: TrainModel):
        super().__init__(category)
        self.encoding_dim = 128
        self.gat_units = 64
        self.num_heads = 4
        self.dense_units = [512, 256]
        self.droprate = 0.3

    def build_model(self, data: TrainingDrugDataDTO, x_train, has_interaction_description: bool = False):

        output_models_1 = []
        output_models_2 = []
        decode_models_1 = []
        decode_models_2 = []
        input_layers_1 = []
        input_layers_2 = []

        gat_layer = GATLayer(units=self.gat_units, num_heads=self.num_heads)

        idx = 0
        while idx < len(data.train_values):
            d = data.train_values[idx]

            if d.category == Category.Substructure:

                smiles_input_shape = x_train[idx][0].shape

                # SMILES input (graph-like input)
                smiles_input_1 = Input(shape=smiles_input_shape, name="Drug1_SMILES_Input")
                smiles_input_2 = Input(shape=smiles_input_shape, name="Drug2_SMILES_Input")
                input_layers_1.append(smiles_input_1)
                input_layers_2.append(smiles_input_2)

                idx += 1

                adjacency_input_shape = x_train[idx][0].shape

                # SMILES input (graph-like input)
                adjacency_input_1 = Input(shape=adjacency_input_shape, name="Drug1_Adjacency_Input")
                adjacency_input_2 = Input(shape=adjacency_input_shape, name="Drug2_Adjacency_Input")
                input_layers_1.append(adjacency_input_1)
                input_layers_2.append(adjacency_input_2)

                # GAT processing for SMILES
                gat_output_1 = gat_layer((smiles_input_1, adjacency_input_1))
                gat_output_1 = ReduceMeanLayer(axis=1)(gat_output_1)  # Aggregate over all nodes
                output_models_1.append(gat_output_1)

                # GAT processing for SMILES
                gat_output_2 = gat_layer((smiles_input_2, adjacency_input_2))
                gat_output_2 = ReduceMeanLayer(axis=1)(gat_output_2)  # Aggregate over all nodes
                output_models_2.append(gat_output_2)

            else:
                autoencoder_layer = AutoEncoderLayer(encoding_dim=self.encoding_dim, input_dim=x_train[idx][0].shape[0])

                input_layer_1 = Input(shape=x_train[idx][0].shape)
                decoded_output_1, encoded_model_1 = autoencoder_layer(input_layer_1)
                input_layers_1.append(input_layer_1)
                output_models_1.append(encoded_model_1)
                decode_models_1.append(decoded_output_1)

                input_layer_2 = Input(shape=x_train[idx][0].shape)
                decoded_output_2, encoded_model_2 = autoencoder_layer(input_layer_2)
                input_layers_2.append(input_layer_2)
                output_models_2.append(encoded_model_2)
                decode_models_2.append(decoded_output_2)

            idx += 1

        decodes_output = decode_models_1 + decode_models_2

        # Concatenate GAT outputs with their respective feature inputs
        combined_drug_1 = Concatenate()(output_models_1)
        combined_drug_2 = Concatenate()(output_models_2)

        # Combine both drugs' information for interaction prediction
        if has_interaction_description:
            autoencoder_layer = AutoEncoderLayer(encoding_dim=self.encoding_dim, input_dim=x_train[idx][0].shape[0])

            input_layer = Input(shape=x_train[idx][0].shape)
            decoded_output, encoded_model = autoencoder_layer(input_layer)

            combined = Concatenate()([combined_drug_1, combined_drug_2, encoded_model])
        else:
            combined = Concatenate()([combined_drug_1, combined_drug_2])

        # Dense layers for final prediction
        train_in = combined
        for units in self.dense_units:
            train_in = Dense(units, activation="relu")(train_in)
            train_in = BatchNormalization()(train_in)
            train_in = Dropout(self.droprate)(train_in)

        # Output layer (example for binary classification, use appropriate activation and units for your case)
        train_in = Dense(self.num_classes)(train_in)
        main_output = Activation('softmax')(train_in)

        model_inputs = input_layers_1 + input_layers_2

        model = Model(inputs=model_inputs, outputs=decodes_output + [main_output])

        return model

    def train(self, parameters: SplitInteractionSimilaritiesTrainingParameterModel) -> (TrainingSummaryDTO, object):

        x_train, x_val, x_test, y_train, y_val, y_test = super().split_deepface_train_val_test(parameters.drug_data, parameters.interaction_data,
                                                                                               train_id=parameters.train_id)

        model = self.build_model(parameters.drug_data[0], x_train, bool(parameters.interaction_data[0].interaction_description))

        return super().fit_dnn_model_multi_model(data_params=DataParams(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test),
                                                 training_params=TrainingParams(train_id=parameters.train_id, optimizer='adam', loss=parameters.loss_function,
                                                                                class_weight=parameters.class_weight),
                                                 multi_task_model=model,
                                                 interactions=parameters.interaction_data)
