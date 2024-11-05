from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Concatenate, Dropout, BatchNormalization, Activation
# noinspection PyUnresolvedReferences
from tensorflow.keras.callbacks import EarlyStopping

from businesses.trains.layers.encoder_layer import EncoderLayer
from businesses.trains.layers.gat_layer import GATLayer
from businesses.trains.layers.reduce_mean_layer import ReduceMeanLayer
from businesses.trains.models.train_base_model import TrainBaseModel
from common.enums.category import Category
from common.helpers import loss_helper
from core.models.training_params import TrainingParams
from core.repository_models.training_drug_interaction_dto import TrainingDrugInteractionDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO


# class GatEncModel(Model):
#     def __init__(self, categories: dict, num_classes=65, use_interaction_description=False):
#
#         super(GatEncModel, self).__init__()
#
#         # Initialize parameters
#         self.encoding_dim = 128
#         self.gat_units = 64
#         self.num_heads = 4
#         self.dense_units = [512, 256]
#         self.droprate = 0.3
#         self.num_classes = num_classes
#         self.use_interaction_description = use_interaction_description
#
#         self.encoder_layers = []
#         for index, category in categories.items():
#             # Define EncoderLayer instances based on the category dictionary
#             if category != Category.Substructure:
#                 self.encoder_layers.append((index, EncoderLayer(encoding_dim=self.encoding_dim)))
#             else:
#                 # Define GAT and ReduceMean layers
#                 self.gat_layer = GATLayer(units=self.gat_units, num_heads=self.num_heads)
#                 self.reduce_mean_layer = ReduceMeanLayer(axis=1)
#
#         # Only create the EncoderLayer for interaction description if needed
#         if self.use_interaction_description:
#             self.interaction_encoder_layer = EncoderLayer(encoding_dim=self.encoding_dim)
#
#         # Define Dense layers for final prediction
#         self.dense_layers = []
#         for units in self.dense_units:
#             self.dense_layers.append(Dense(units, activation="relu"))
#             self.dense_layers.append(BatchNormalization())
#             self.dense_layers.append(Dropout(self.droprate))
#
#         # Define output layer
#         self.output_layer = Dense(self.num_classes, activation='softmax')
#
#     def call(self, inputs):
#         input_layers_1, input_layers_2 = inputs[:len(inputs) // 2], inputs[len(inputs) // 2:]
#
#         output_models_1 = []
#         output_models_2 = []
#
#         idx = 0
#
#         while idx < len(input_layers_1):
#             if any(e[0] == idx for e in self.encoder_layers):  # Check if EncoderLayer exists for this index
#                 # Get the corresponding EncoderLayer instance for this index
#                 encoder_layer = next(e[1] for e in self.encoder_layers if e[0] == idx)
#
#                 # Process Drug 1
#                 input_layer_1 = input_layers_1[idx]
#                 encoded_model_1 = encoder_layer(input_layer_1)
#                 output_models_1.append(encoded_model_1)
#
#                 # Process Drug 2
#                 input_layer_2 = input_layers_2[idx]
#                 encoded_model_2 = encoder_layer(input_layer_2)
#                 output_models_2.append(encoded_model_2)
#
#                 idx += 1  # Move to the next input
#
#             else:
#                 # GAT processing for Drug 1
#                 smiles_input_1, adjacency_input_1 = input_layers_1[idx], input_layers_1[idx + 1]
#                 smiles_input_2, adjacency_input_2 = input_layers_2[idx], input_layers_2[idx + 1]
#
#                 gat_output_1 = self.gat_layer((smiles_input_1, adjacency_input_1))
#                 gat_output_1 = self.reduce_mean_layer(gat_output_1)
#                 output_models_1.append(gat_output_1)
#
#                 gat_output_2 = self.gat_layer((smiles_input_2, adjacency_input_2))
#                 gat_output_2 = self.reduce_mean_layer(gat_output_2)
#                 output_models_2.append(gat_output_2)
#
#                 idx += 2  # Move to the next pair of inputs
#
#         # Concatenate GAT and encoded outputs with their respective feature inputs
#         combined_drug_1 = Concatenate()(output_models_1)
#         combined_drug_2 = Concatenate()(output_models_2)
#
#         # Combine both drugs' information for interaction prediction
#         if self.use_interaction_description:
#             # Process the interaction description input
#             encoded_model = self.interaction_encoder_layer(inputs[-1])
#             combined = Concatenate()([combined_drug_1, combined_drug_2, encoded_model])
#         else:
#             combined = Concatenate()([combined_drug_1, combined_drug_2])
#
#         # Pass through Dense layers for final prediction
#         x = combined
#         for dense_layer in self.dense_layers:
#             x = dense_layer(x)
#
#         # Output layer
#         return self.output_layer(x)
#
#     def compile_model(self, optimizer='adam', loss='categorical_crossentropy', metrics=None):
#         if metrics is None:
#             metrics = ['accuracy']
#
#         self.compile(optimizer=optimizer, loss=loss, metrics=metrics)


class GatEncTrainModel(TrainBaseModel):
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
        input_layers_1 = []
        input_layers_2 = []

        gat_layer = GATLayer(units=self.gat_units, num_heads=self.num_heads)
        reduce_mean_layer = ReduceMeanLayer(axis=1)

        for idx, category in data_categories.items():

            if data_categories[idx] == Category.Substructure:

                smiles_input_shape = x_train_shapes[idx]

                # SMILES input (graph-like input)
                smiles_input_1 = Input(shape=smiles_input_shape, name="Drug1_SMILES_Input")
                smiles_input_2 = Input(shape=smiles_input_shape, name="Drug2_SMILES_Input")
                input_layers_1.append(smiles_input_1)
                input_layers_2.append(smiles_input_2)

                adjacency_input_shape = x_train_shapes[idx + 1]

                # SMILES input (graph-like input)
                adjacency_input_1 = Input(shape=adjacency_input_shape, name="Drug1_Adjacency_Input")
                adjacency_input_2 = Input(shape=adjacency_input_shape, name="Drug2_Adjacency_Input")
                input_layers_1.append(adjacency_input_1)
                input_layers_2.append(adjacency_input_2)

                # GAT processing for SMILES
                gat_output_1 = gat_layer((smiles_input_1, adjacency_input_1))
                gat_output_1 = reduce_mean_layer(gat_output_1)  # Aggregate over all nodes
                output_models_1.append(gat_output_1)

                # GAT processing for SMILES
                gat_output_2 = gat_layer((smiles_input_2, adjacency_input_2))
                gat_output_2 = reduce_mean_layer(gat_output_2)  # Aggregate over all nodes
                output_models_2.append(gat_output_2)

            else:
                encoder_layer = EncoderLayer(encoding_dim=self.encoding_dim)

                input_layer_1 = Input(shape=x_train_shapes[idx])
                encoded_model_1 = encoder_layer(input_layer_1)
                input_layers_1.append(input_layer_1)
                output_models_1.append(encoded_model_1)

                input_layer_2 = Input(shape=x_train_shapes[idx])
                encoded_model_2 = encoder_layer(input_layer_2)
                input_layers_2.append(input_layer_2)
                output_models_2.append(encoded_model_2)

        # Concatenate GAT outputs with their respective feature inputs
        combined_drug_1 = Concatenate()(output_models_1)
        combined_drug_2 = Concatenate()(output_models_2)

        # Combine both drugs' information for interaction prediction
        if has_interaction_description:
            encoder_layer = EncoderLayer(encoding_dim=self.encoding_dim)

            input_layer = Input(shape=x_train_shapes[-1])
            encoded_model = encoder_layer(input_layer)

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
        output = Activation('softmax')(train_in)

        model_inputs = input_layers_1 + input_layers_2

        model = Model(inputs=model_inputs, outputs=output, name="GatEncModel")

        return model

    def fit_model(self, x_train, y_train, x_val, y_val, x_test, y_test) -> TrainingSummaryDTO:

        if self.training_params.class_weight:
            class_weights = loss_helper.get_class_weights(y_train)
        else:
            class_weights = None

        x_train_shapes = [x[0].shape for x in x_train]

        model = self.build_model(self.categories, x_train_shapes, bool(self.interaction_data[0].interaction_description))
        model.compile(optimizer=self.training_params.optimizer,
                      loss=loss_helper.get_loss_function(self.training_params.loss, class_weights),
                      metrics=self.training_params.metrics)

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

        history = model.fit(x_train, y_train, epochs=self.training_params.epoch_num, batch_size=128, validation_data=(x_val, y_val), callbacks=early_stopping)

        self.plot_accuracy(history, f"{self.train_id}")
        self.plot_loss(history, f"{self.train_id}")

        y_pred = model.predict(x_test)

        result = self.calculate_evaluation_metrics(model, x_test, y_test, y_pred=y_pred)

        result.model_info = self.get_model_info(model)

        if self.interaction_data is not None:
            result.data_report = self.get_data_report_split(self.interaction_data, y_train, y_test)

        return result
