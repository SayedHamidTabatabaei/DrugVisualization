# noinspection PyUnresolvedReferences
from tensorflow.keras import Model, Input
# noinspection PyUnresolvedReferences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Concatenate, Dropout, BatchNormalization, Activation

from businesses.trains.layers.autoencoder_layer import AutoEncoderLayer
from businesses.trains.layers.gat_layer import GATLayer
from businesses.trains.layers.reduce_mean_layer import ReduceMeanLayer
from businesses.trains.models.train_base_model import TrainBaseModel
from common.enums.category import Category
from common.helpers import loss_helper
from core.models.training_params import TrainingParams
from core.repository_models.training_drug_interaction_dto import TrainingDrugInteractionDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO


class GatAeTrainModel(TrainBaseModel):
    def __init__(self, train_id: int, categories: dict, num_classes: int, interaction_data: list[TrainingDrugInteractionDTO], training_params: TrainingParams):
        super().__init__(train_id, num_classes)
        self.categories = categories
        self.interaction_data = interaction_data
        self.training_params = training_params
        self.encoding_dim = 128
        self.gat_units = 64
        self.num_heads = 4
        self.dense_units = [512, 256]
        self.droprate = 0.3

    def build_model(self, data_categories: dict, x_train_shapes, has_interaction_description: bool = False):

        output_models_1 = []
        output_models_2 = []
        decode_models_1 = []
        decode_models_2 = []
        input_layers_1 = []
        input_layers_2 = []
        input_layer_int = None

        gat_layer = GATLayer(units=self.gat_units, num_heads=self.num_heads)
        reduce_mean_layer = ReduceMeanLayer(axis=1)

        for idx, category in data_categories.items():

            if category == Category.Substructure:
                smiles_input_shape = x_train_shapes[idx]
                adjacency_input_shape = x_train_shapes[idx + 1]

                smiles_input_1, smiles_input_2, adjacency_input_1, adjacency_input_2, gat_output_1, gat_output_2 = \
                    self.build_gat_layer(gat_layer, reduce_mean_layer, smiles_input_shape, adjacency_input_shape)

                input_layers_1.append(smiles_input_1)
                input_layers_2.append(smiles_input_2)

                input_layers_1.append(adjacency_input_1)
                input_layers_2.append(adjacency_input_2)

                output_models_1.append(gat_output_1)
                output_models_2.append(gat_output_2)

            else:
                autoencoder_layer = AutoEncoderLayer(encoding_dim=self.encoding_dim, input_dim=x_train_shapes[idx][0])

                input_layer_1 = Input(shape=x_train_shapes[idx], name=f"AE_Input_1_{idx}")
                decoded_output_1, encoded_model_1 = autoencoder_layer(input_layer_1)
                input_layers_1.append(input_layer_1)
                output_models_1.append(encoded_model_1)
                decode_models_1.append(decoded_output_1)

                input_layer_2 = Input(shape=x_train_shapes[idx], name=f"AE_Input_2_{idx}")
                decoded_output_2, encoded_model_2 = autoencoder_layer(input_layer_2)
                input_layers_2.append(input_layer_2)
                output_models_2.append(encoded_model_2)
                decode_models_2.append(decoded_output_2)

        # Concatenate GAT outputs with their respective feature inputs
        combined_drug_1 = Concatenate()(output_models_1)
        combined_drug_2 = Concatenate()(output_models_2)

        # Combine both drugs' information for interaction prediction
        if has_interaction_description:
            autoencoder_layer = AutoEncoderLayer(encoding_dim=self.encoding_dim, input_dim=x_train_shapes[-1][0])

            input_layer = Input(shape=x_train_shapes[-1])
            decoded_output, encoded_model = autoencoder_layer(input_layer)
            input_layer_int = input_layer

            decodes_output = decode_models_1 + decode_models_2 + [decoded_output]
            combined = Concatenate()([combined_drug_1, combined_drug_2, encoded_model])
        else:
            decodes_output = decode_models_1 + decode_models_2
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

        if input_layer_int:
            model_inputs = input_layers_1 + input_layers_2 + [input_layer_int]
        else:
            model_inputs = input_layers_1 + input_layers_2

        model = Model(inputs=model_inputs, outputs=decodes_output + [main_output], name="GAT_AE")

        return model

    def compile_model(self, model):
        # List of loss functions for each output
        num_autoencoders = len(model.outputs) - 1  # Last output is for classification

        losses = ['mse'] * num_autoencoders + [loss_helper.get_loss_function(self.training_params.loss)]

        # Loss weights: You can adjust these to give more importance to classification vs. autoencoder loss
        loss_weights = [1.0] * num_autoencoders + [1.0]

        # List of metrics for each output
        metrics = ['mse'] * num_autoencoders + ['accuracy']

        # Compile the model with separate losses and metrics for each output
        model.compile(
            optimizer='adam',
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )

        return model

    def generate_y_data_autoencoder(self, x_data, y_data):

        data_type_count = len(x_data) // 2
        parallel_autoencoder_indexes = [idx for idx, category in self.categories.items() if category != Category.Substructure]
        parallel_autoencoder_indexes = parallel_autoencoder_indexes + [i + data_type_count for i in parallel_autoencoder_indexes]

        y = [x for idx, x in enumerate(x_data) if idx in parallel_autoencoder_indexes] + [y_data]

        return y

    def fit_model(self, x_train, y_train, x_val, y_val, x_test, y_test) -> TrainingSummaryDTO:

        x_train_shapes = [x[0].shape for x in x_train]

        train_dataset, val_dataset, test_dataset, train_generator_length, val_generator_length, test_generator_length = (
            self.big_data_loader(x_train, y_train, x_val, y_val, x_test, y_test))

        model = self.build_model(self.categories, x_train_shapes, bool(self.interaction_data[0].interaction_description))

        model = self.compile_model(model)

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

        print('Fit data!')
        history = model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=[early_stopping],
                            steps_per_epoch=train_generator_length, validation_steps=val_generator_length)

        self.save_plots(history, f"{self.train_id}")

        y_pred = model.predict(test_dataset, steps=test_generator_length)
        y_pred = y_pred[-1]

        result = self.calculate_evaluation_metrics(model, x_test, y_test, y_pred=y_pred)

        result.model_info = self.get_model_info(model)

        if self.interaction_data is not None:
            result.data_report = self.get_data_report_split(self.interaction_data, y_train, y_test)

        return result
