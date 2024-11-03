from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization, Activation
# noinspection PyUnresolvedReferences
from tensorflow.keras.models import Model
# noinspection PyUnresolvedReferences
from tensorflow.keras.callbacks import EarlyStopping

from businesses.trains.layers.encoder_layer import EncoderLayer
from businesses.trains.layers.gat_layer import GATLayer
from businesses.trains.layers.reduce_mean_layer import ReduceMeanLayer
from businesses.trains.train_base_service import TrainBaseService
from common.enums.category import Category
from common.enums.train_models import TrainModel
from core.models.data_params import DataParams
from core.models.training_parameter_models.split_drugs_test_with_train_training_parameter_model import SplitDrugsTestWithTrainTrainingParameterModel
from core.models.training_params import TrainingParams
from core.repository_models.training_drug_data_dto import TrainingDrugDataDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.Drug_GAT_Enc_Con_DNN


class DrugGatEncConDnnTrainService(TrainBaseService):

    def __init__(self, category):
        super().__init__(category)
        self.encoding_dim = 128
        self.gat_units = 64
        self.num_heads = 4
        self.dense_units = [512, 256]
        self.droprate = 0.3

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
                encoder_layer = EncoderLayer(encoding_dim=self.encoding_dim)

                input_layer_1 = Input(shape=x_train[idx][0].shape)
                encoded_model_1 = encoder_layer(input_layer_1)
                input_layers_1.append(input_layer_1)
                encoded_models_1.append(encoded_model_1)

                input_layer_2 = Input(shape=x_train[idx][0].shape)
                encoded_model_2 = encoder_layer(input_layer_2)
                input_layers_2.append(input_layer_2)
                encoded_models_2.append(encoded_model_2)

        # Concatenate GAT outputs with their respective feature inputs
        combined_drug_1 = Concatenate()([gat_output_1] + encoded_models_1)
        combined_drug_2 = Concatenate()([gat_output_2] + encoded_models_2)

        # Combine both drugs' information for interaction prediction
        combined = Concatenate()([combined_drug_1, combined_drug_2])

        # Dense layers for final prediction
        train_in = combined
        for units in self.dense_units:
            train_in = Dense(units, activation="relu")(train_in)
            train_in = BatchNormalization()(train_in)
            train_in = Dropout(self.droprate)(train_in)

        # Output layer (example for binary classification, use appropriate activation and units for your case)
        train_in = Dense(self.num_classes)(train_in)
        output = Activation('softmax')(train_in)

        model_inputs = [smiles_input_1] + input_layers_1 + [smiles_input_2] + input_layers_2

        model = Model(inputs=model_inputs, outputs=output)

        return model

    def train(self, parameters: SplitDrugsTestWithTrainTrainingParameterModel) -> TrainingSummaryDTO:

        results = []

        for x_train, x_test, y_train, y_test in super().manual_k_fold_train_test_data(parameters.drug_data, parameters.interaction_data, is_deep_face=True):
            model = self.build_model(parameters.drug_data[0], x_train)

            result = super().fit_dnn_model(data_params=DataParams(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test),
                                           training_params=TrainingParams(train_id=parameters.train_id, optimizer='adam', loss=parameters.loss_function,
                                                                          class_weight=parameters.class_weight),
                                           model=model,
                                           interactions=parameters.interaction_data)

            results.append(result)

        return super().calculate_fold_results(results)
