import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Concatenate, Dropout, Reshape, Flatten, MultiHeadAttention, BatchNormalization, Activation, Lambda
from tensorflow.keras.metrics import MeanSquaredError, Accuracy

from businesses.trains.helpers.customize_epoch_progress import CustomizeEpochProgress
from businesses.trains.layers.autoencoder_layer import AutoEncoderLayer
from businesses.trains.layers.autoencoder_multi_dim_layer import AutoEncoderMultiDimLayer
from businesses.trains.layers.gat_layer import GATLayer
from businesses.trains.layers.reduce_mean_layer import ReduceMeanLayer
from businesses.trains.models.train_base_model import TrainBaseModel
from common.enums.category import Category
from common.helpers import loss_helper
from core.models.training_params import TrainingParams
from core.repository_models.training_drug_interaction_dto import TrainingDrugInteractionDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO


class GatMhaRDTrainModel(TrainBaseModel):
    def __init__(self, train_id: int, categories: dict, num_classes: int, interaction_data: list[TrainingDrugInteractionDTO],
                 training_params: TrainingParams, reverse: bool = False):
        super().__init__(train_id, num_classes)
        self.categories = categories
        self.interaction_data = interaction_data
        self.training_params = training_params
        self.reverse = reverse
        self.str_encoding_shape = (24, 16)
        self.encoding_dim = self.str_encoding_shape[1]
        self.gat_units = 64
        self.gat_num_heads = 4
        self.mha_num_heads = 8
        self.dense_units = [512, 256]
        self.droprate = 0.3

    def build_model(self, x_train_shapes, has_interaction_description: bool = False):

        input_layers_1 = []
        input_layers_2 = []
        input_layer_int = None
        decode_models_1 = []
        decode_models_2 = []
        decode_2d_models_1 = []
        decode_2d_models_2 = []
        output_models_1 = []
        output_models_2 = []

        sim_encoded_models_1 = []
        sim_encoded_models_2 = []
        str_encoded_models_1 = []
        str_encoded_models_2 = []

        gat_layer = GATLayer(units=self.gat_units, num_heads=self.gat_num_heads)
        reduce_mean_layer = ReduceMeanLayer(axis=1)

        for idx, category in self.categories.items():

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

            elif category.data_type == str:
                autoencoder_layer = AutoEncoderMultiDimLayer(encoding_dim=self.str_encoding_shape, target_shape=x_train_shapes[idx])

                str_input_layer_1 = Input(shape=x_train_shapes[idx], name=f"AE_Str_Input_1_{idx}")
                decoded_output_1, encoded_model_1 = autoencoder_layer(str_input_layer_1)
                input_layers_1.append(str_input_layer_1)
                str_encoded_models_1.append(encoded_model_1)
                decode_2d_models_1.append(decoded_output_1)

                str_input_layer_2 = Input(shape=x_train_shapes[idx], name=f"AE_Str_Input_2_{idx}")
                decoded_output_2, encoded_model_2 = autoencoder_layer(str_input_layer_2)
                input_layers_2.append(str_input_layer_2)
                str_encoded_models_2.append(encoded_model_2)
                decode_2d_models_2.append(decoded_output_2)

            else:
                autoencoder_layer = AutoEncoderLayer(encoding_dim=self.encoding_dim, input_dim=x_train_shapes[idx][0])

                sim_input_layer_1 = Input(shape=x_train_shapes[idx], name=f"AE_Sim_Input_1_{idx}")
                decoded_output_1, encoded_model_1 = autoencoder_layer(sim_input_layer_1)
                input_layers_1.append(sim_input_layer_1)
                sim_encoded_models_1.append(encoded_model_1)
                decode_models_1.append(decoded_output_1)

                sim_input_layer_2 = Input(shape=x_train_shapes[idx], name=f"AE_Sim_Input_2_{idx}")
                decoded_output_2, encoded_model_2 = autoencoder_layer(sim_input_layer_2)
                input_layers_2.append(sim_input_layer_2)
                sim_encoded_models_2.append(encoded_model_2)
                decode_models_2.append(decoded_output_2)

        joint_encoders_1 = Lambda(lambda x: tf.stack(x, axis=1))(sim_encoded_models_1)
        joint_encoders_2 = Lambda(lambda x: tf.stack(x, axis=1))(sim_encoded_models_2)

        text_attentions_1 = []
        text_attentions_2 = []

        for i in range(len(str_encoded_models_1)):

            attention_layer = MultiHeadAttention(num_heads=self.mha_num_heads, key_dim=64, name=f"MultiHeadAttention_{i}")

            attention_output_1 = self.generate_multi_head_attention(attention_layer, query_parameter=joint_encoders_1,
                                                                    key_value_parameter=str_encoded_models_1[i], reverse=self.reverse)

            attention_output_2 = self.generate_multi_head_attention(attention_layer, query_parameter=joint_encoders_2,
                                                                    key_value_parameter=str_encoded_models_2[i], reverse=self.reverse)

            text_attentions_1.append(attention_output_1)
            text_attentions_2.append(attention_output_2)

        final_text_attention_1, final_text_attention_2 = self.reduce_multiple_attentions(text_attentions_1, text_attentions_2)

        flatten = Flatten(name=f"Flatten")
        final_text_attention_1 = flatten(final_text_attention_1)
        final_text_attention_2 = flatten(final_text_attention_2)

        output_models_1.append(final_text_attention_1)
        output_models_2.append(final_text_attention_2)

        combined_drug_1 = Concatenate(name="CombinedAttentionOutput_1")(output_models_1)
        combined_drug_2 = Concatenate(name="CombinedAttentionOutput_2")(output_models_2)

        # Combine both drugs' information for interaction prediction
        if has_interaction_description:
            autoencoder_layer = AutoEncoderLayer(encoding_dim=self.encoding_dim, input_dim=x_train_shapes[-1][0])

            input_layer_str = Input(shape=x_train_shapes[-1])
            decoded_output, encoded_model = autoencoder_layer(input_layer_str)
            input_layer_int = input_layer_str

            attention_outputs = []
            for j in range(len(sim_encoded_models_1)):
                attention_layer = MultiHeadAttention(num_heads=self.mha_num_heads, key_dim=64, name=f"MultiHeadAttention_Int_{j}")
                flatten = Flatten(name=f"Flatten_Int_{j}")

                reshaped_other_input = Reshape((1, sim_encoded_models_1[j].shape[-1]))(sim_encoded_models_1[j])
                attention_output = self.generate_multi_head_attention(attention_layer,
                                                                      query_parameter=reshaped_other_input,
                                                                      key_value_parameter=input_layer_str,
                                                                      reverse=self.reverse)
                attention_output = flatten(attention_output)
                attention_outputs.append(attention_output)

            decodes_output = decode_models_1 + decode_models_2 + [decoded_output]
            decodes_2d_output = decode_2d_models_1 + decode_2d_models_2

            combined_drug_int = Concatenate(name="CombinedAttentionOutput_Int")(attention_outputs)
            combined = Concatenate()([combined_drug_1, combined_drug_2, combined_drug_int])

        else:
            decodes_output = decode_models_1 + decode_models_2
            decodes_2d_output = decode_2d_models_1 + decode_2d_models_2
            combined = Concatenate()([combined_drug_1, combined_drug_2])

        train_in = combined
        for units in self.dense_units:
            train_in = Dense(units, activation="relu")(train_in)
            train_in = BatchNormalization()(train_in)
            train_in = Dropout(self.droprate)(train_in)

        train_in = Dense(self.num_classes)(train_in)
        main_output = Activation('softmax')(train_in)

        if input_layer_int:
            model_inputs = input_layers_1 + input_layers_2 + [input_layer_int]
        else:
            model_inputs = input_layers_1 + input_layers_2

        model = Model(inputs=model_inputs, outputs=decodes_output + decodes_2d_output + [main_output], name="GAT_MHA")

        return model

    @staticmethod
    def generate_multi_head_attention(attention_layer, query_parameter, key_value_parameter, reverse: bool = False):

        if reverse:
            item = query_parameter
            query_parameter = key_value_parameter
            key_value_parameter = item

        attention_output = attention_layer(query=query_parameter, key=key_value_parameter, value=key_value_parameter)

        return attention_output

    def reduce_multiple_attentions(self, attentions_1, attentions_2):
        assert len(attentions_1) == len(attentions_2), "Attention layers must have same number of attention layers"

        if len(attentions_1) == 1:
            return attentions_1[0], attentions_2[0]

        output_models_1 = []
        output_models_2 = []

        query_attention_1 = attentions_1[0]
        query_attention_2 = attentions_2[0]

        for j in range(1, len(attentions_1)):
            attention_layer = MultiHeadAttention(num_heads=8, key_dim=64, name=f"MultiHeadAttention_{len(attentions_1)}_{j}")

            attention_output = self.generate_multi_head_attention(attention_layer, query_parameter=query_attention_1, key_value_parameter=attentions_1[j])
            output_models_1.append(attention_output)

            attention_output = self.generate_multi_head_attention(attention_layer, query_parameter=query_attention_2, key_value_parameter=attentions_2[j])
            output_models_2.append(attention_output)

        return self.reduce_multiple_attentions(output_models_1, output_models_2)

    def compile_model(self, model):

        def custom_mse_3d(y_true, y_pred):
            # Flatten both true and predicted values across the 3D dimensions
            y_true = tf.reshape(y_true, [-1])
            y_pred = tf.reshape(y_pred, [-1])
            return tf.reduce_mean(tf.square(y_true - y_pred))

        # List of loss functions for each output
        num_autoencoders = len(model.outputs) - 1  # Last output is for classification

        decodes_1d_outputs = []
        decodes_2d_outputs = []

        for decode_output in model.outputs[:num_autoencoders]:
            if len(decode_output.shape) == 2:  # 2D output (None, 936)
                decodes_1d_outputs.append(decode_output)
            else:  # 3D output (None, 768, 512)
                decodes_2d_outputs.append(decode_output)

        # Losses for 1D (MSE)
        losses_1d = ['mse' for _ in range(len(decodes_1d_outputs))]

        # Custom loss for 2D (Pixel-wise MSE or another loss)
        losses_2d = [custom_mse_3d for _ in range(len(decodes_2d_outputs))]  # You can use your custom loss for 3D data

        # Combine losses
        losses = losses_1d + losses_2d + [loss_helper.get_loss_function(self.training_params.loss)]

        # Loss weights (adjust as needed)
        loss_weights = [1.0] * len(decodes_1d_outputs) + [1.0] * len(decodes_2d_outputs) + [1.0]

        # Metrics
        metrics = ([MeanSquaredError(name=f'auto_encoder_{i}_mse') for i in range(len(decodes_1d_outputs))] +
                   [MeanSquaredError(name=f'auto_encoder_2d_{i}_mse') for i in range(len(decodes_2d_outputs))] +
                   [Accuracy(name='classification_accuracy')])

        # Compile model with these separate losses and metrics
        model.compile(optimizer='adam', loss=losses, loss_weights=loss_weights, metrics=metrics)

        return model

    def generate_y_data_autoencoder(self, x_data, y_data):

        data_type_count = len(x_data) // 2
        parallel_autoencoder_indexes = [idx for idx, category in self.categories.items()
                                        if category != Category.Substructure and category.data_type != str]
        parallel_autoencoder_indexes = (parallel_autoencoder_indexes +
                                        [i + data_type_count for i in parallel_autoencoder_indexes])

        parallel_autoencoder_multi_dim_indexes = [idx for idx, category in self.categories.items()
                                                  if category.data_type == str]
        parallel_autoencoder_multi_dim_indexes = (parallel_autoencoder_multi_dim_indexes +
                                                  [i + data_type_count for i in parallel_autoencoder_multi_dim_indexes])

        y = ([x for idx, x in enumerate(x_data) if idx in parallel_autoencoder_indexes] +
             [x for idx, x in enumerate(x_data) if idx in parallel_autoencoder_multi_dim_indexes] +
             [y_data])

        return y

    def fit_model(self, x_train, y_train, x_val, y_val, x_test, y_test) -> TrainingSummaryDTO:

        x_train_shapes = [x[0].shape for x in x_train]

        train_dataset, val_dataset, test_dataset, train_generator_length, val_generator_length, test_generator_length = (
            self.big_data_loader(x_train, y_train, x_val, y_val, x_test, y_test, multiple_output=True))

        model = self.build_model(x_train_shapes, bool(self.interaction_data[0].interaction_description))

        model = self.compile_model(model)

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

        progress_bar_with_epoch_logging = CustomizeEpochProgress()

        callbacks = [early_stopping, progress_bar_with_epoch_logging]

        print('Fit data!')
        history = model.fit(train_dataset, epochs=10, validation_data=val_dataset,
                            steps_per_epoch=train_generator_length, validation_steps=val_generator_length,
                            verbose=0, callbacks=callbacks)

        self.save_plots(history, self.train_id)

        y_pred = model.predict(test_dataset, steps=test_generator_length)
        y_pred = y_pred[-1]

        result = self.calculate_evaluation_metrics(model, x_test, y_test, y_pred=y_pred)

        result.model_info = self.get_model_info(model)

        if self.interaction_data is not None:
            result.data_report = self.get_data_report_split(self.interaction_data, y_train, y_test)

        return result
