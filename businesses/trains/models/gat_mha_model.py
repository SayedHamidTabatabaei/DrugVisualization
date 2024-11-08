import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
# noinspection PyUnresolvedReferences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Concatenate, Dropout, Reshape, Flatten, MultiHeadAttention, BatchNormalization, Activation

from businesses.trains.layers.data_generator import DataGenerator
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


class GatMhaTrainModel(TrainBaseModel):
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

    def get_output_signature(self, x_train):
        signature = []
        for i, dataset in enumerate(x_train):
            # Get the shape based on the first element in the list (assuming all elements are the same shape within each sub-list)
            first_element_shape = (None,) + tuple(np.array(dataset[0]).shape)
            signature.append(tf.TensorSpec(shape=first_element_shape, dtype=tf.float32))

        return tuple(signature), tf.TensorSpec(shape=(None, self.num_classes), dtype=tf.float32)

    @staticmethod
    def build_gat_layer(gat_layer, reduce_mean_layer, smiles_input_shape, adjacency_input_shape):
        smiles_input_1 = Input(shape=smiles_input_shape, name="Drug1_SMILES_Input")
        smiles_input_2 = Input(shape=smiles_input_shape, name="Drug2_SMILES_Input")

        adjacency_input_1 = Input(shape=adjacency_input_shape, name="Drug1_Adjacency_Input")
        adjacency_input_2 = Input(shape=adjacency_input_shape, name="Drug2_Adjacency_Input")

        gat_output_1 = gat_layer((smiles_input_1, adjacency_input_1))
        gat_output_1 = reduce_mean_layer(gat_output_1)

        gat_output_2 = gat_layer((smiles_input_2, adjacency_input_2))
        gat_output_2 = reduce_mean_layer(gat_output_2)

        return smiles_input_1, smiles_input_2, adjacency_input_1, adjacency_input_2, gat_output_1, gat_output_2

    def build_model(self, data_categories: dict, x_train_shapes, has_interaction_description: bool = False):

        input_layers_1 = []
        input_layers_2 = []
        output_models_1 = []
        output_models_2 = []

        input_layers_str_1 = []
        input_layers_str_2 = []
        input_layers_other_1 = []
        input_layers_other_2 = []

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

            elif category.data_type == 'str':
                input_layer_str_1 = Input(shape=x_train_shapes[idx], name=f"Str_Input_1_{idx}")
                input_layers_str_1.append(input_layer_str_1)

                second_idx = idx + (len(x_train_shapes) // 2)

                input_layer_str_2 = Input(shape=x_train_shapes[second_idx], name=f"Str_Input_2_{second_idx}")
                input_layers_str_2.append(input_layer_str_2)

                input_layers_1.append(input_layer_str_1)
                input_layers_2.append(input_layer_str_2)
            else:
                other_input_layer_1 = Input(shape=x_train_shapes[idx], name=f"Other_Input_1_{idx}")
                input_layers_other_1.append(other_input_layer_1)
                input_layers_1.append(other_input_layer_1)

                other_input_layer_2 = Input(shape=x_train_shapes[idx], name=f"Other_Input_2_{idx}")
                input_layers_other_2.append(other_input_layer_2)
                input_layers_2.append(other_input_layer_2)

        for i, str_layer in enumerate(input_layers_str_1):
            for j, other_layer in enumerate(input_layers_other_1):
                attention_layer = MultiHeadAttention(num_heads=8, key_dim=64, name=f"MultiHeadAttention_{i}_{j}")
                flatten = Flatten(name=f"Flatten_{i}_{j}")

                reshaped_other_input = Reshape((1, input_layers_other_1[j].shape[-1]))(input_layers_other_1[j])
                attention_output = attention_layer(query=input_layers_str_1[i], key=reshaped_other_input, value=reshaped_other_input)
                attention_output = flatten(attention_output)
                output_models_1.append(attention_output)

                reshaped_other_input = Reshape((1, input_layers_other_2[j].shape[-1]))(input_layers_other_2[j])
                attention_output = attention_layer(query=input_layers_str_2[i], key=reshaped_other_input, value=reshaped_other_input)
                attention_output = flatten(attention_output)
                output_models_2.append(attention_output)

        combined_drug_1 = Concatenate(name="CombinedAttentionOutput_1")(output_models_1)
        combined_drug_2 = Concatenate(name="CombinedAttentionOutput_2")(output_models_2)

        combined = Concatenate()([combined_drug_1, combined_drug_2])

        train_in = combined
        for units in self.dense_units:
            train_in = Dense(units, activation="relu")(train_in)
            train_in = BatchNormalization()(train_in)
            train_in = Dropout(self.droprate)(train_in)

        train_in = Dense(self.num_classes)(train_in)
        output = Activation('softmax')(train_in)

        model_inputs = input_layers_1 + input_layers_2
        model = Model(inputs=model_inputs, outputs=output, name="GAT_MHA")

        return model

    def fit_model(self, x_train, y_train, x_val, y_val, x_test, y_test) -> TrainingSummaryDTO:

        # if self.training_params.class_weight:
        #     class_weights = loss_helper.get_class_weights(y_train)
        # else:
        #     class_weights = None

        x_train_shapes = [x[0].shape for x in x_train]

        train_generator = DataGenerator(x_train, y_train, batch_size=32, drop_remainder=True)
        val_generator = DataGenerator(x_val, y_val, batch_size=32, drop_remainder=True)
        test_generator = DataGenerator(x_test, y_test, batch_size=1, drop_remainder=True)

        model = self.build_model(self.categories, x_train_shapes, bool(self.interaction_data[0].interaction_description))

        output_signature = self.get_output_signature(x_train)

        train_dataset = tf.data.Dataset.from_generator(lambda: iter(train_generator), output_signature=output_signature).repeat()
        val_dataset = tf.data.Dataset.from_generator(lambda: iter(val_generator), output_signature=output_signature).repeat()
        test_dataset = tf.data.Dataset.from_generator(lambda: iter(test_generator), output_signature=output_signature).repeat()

        model.compile(optimizer='adam', loss=loss_helper.get_loss_function(self.training_params.loss), metrics=['accuracy'])

        steps_per_epoch = len(train_generator)
        validation_steps = len(val_generator)

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

        print('Fit data!')
        history = model.fit(train_dataset, epochs=1, validation_data=val_dataset, callbacks=early_stopping,
                            steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

        self.plot_accuracy(history, f"{self.train_id}")
        self.plot_loss(history, f"{self.train_id}")

        y_pred = model.predict(test_dataset, steps=len(test_generator))

        result = self.calculate_evaluation_metrics(model, x_test, y_test, y_pred=y_pred)

        result.model_info = self.get_model_info(model)

        if self.interaction_data is not None:
            result.data_report = self.get_data_report_split(self.interaction_data, y_train, y_test)

        return result
