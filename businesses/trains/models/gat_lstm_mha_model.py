import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Concatenate, Dropout, Flatten, MultiHeadAttention, \
    BatchNormalization, Activation, Lambda, LSTM, RepeatVector

from businesses.trains.layers.encoder_layer import EncoderLayer
from businesses.trains.layers.gat_layer import GATLayer
from businesses.trains.layers.reduce_pooling_layer import ReducePoolingLayer
from businesses.trains.models.train_base_model import TrainBaseModel
from common.enums.category import Category
from common.enums.similarity_type import SimilarityType
from common.helpers import loss_helper
from core.models.training_params import TrainingParams
from core.repository_models.training_drug_interaction_dto import TrainingDrugInteractionDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO


class GatLstmMhaTrainModel(TrainBaseModel):
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

    def build_model(self, x_train_shapes):

        input_layers_1 = []
        input_layers_2 = []
        input_layer_int = None
        output_models_1 = []
        output_models_2 = []

        sim_encoded_models_1 = []
        sim_encoded_models_2 = []
        str_encoded_models_1 = []
        str_encoded_models_2 = []

        gat_layer = GATLayer(units=self.gat_units, num_heads=self.gat_num_heads)
        reduce_mean_layer = ReducePoolingLayer(axis=1, pooling_mode='mean')

        for idx, category in self.categories.items():

            if category == (Category.Substructure, SimilarityType.Original):
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
            elif category[0].data_type == str:
                input_dim = x_train_shapes[idx][1]  # 512 or the input feature size
                target_sequence_length = self.str_encoding_shape[0]  # 24
                target_feature_dim = self.str_encoding_shape[1]  # 16

                dense_layer = Dense(target_feature_dim)  # Reduces feature dimension to 16
                repeat_vector_layer = RepeatVector(target_sequence_length)  # Repeat the feature vector over 24 time steps
                lstm_layer = LSTM(target_feature_dim, return_sequences=True)  # LSTM layer
                flat_layer = Flatten()

                # First input: Process the first drug
                str_input_layer_1 = Input(shape=(input_dim,), name=f"Enc_Str_Input_1_{idx}")
                dense_output_1 = dense_layer(str_input_layer_1)
                repeated_output_1 = repeat_vector_layer(dense_output_1)  # Repeat the dense output to form a sequence
                lstm_model_1 = lstm_layer(repeated_output_1)
                lstm_model_flat_1 = flat_layer(lstm_model_1)
                input_layers_1.append(str_input_layer_1)
                str_encoded_models_1.append(lstm_model_flat_1)

                # Second input: Process the second drug
                str_input_layer_2 = Input(shape=(input_dim,), name=f"Enc_Str_Input_2_{idx}")
                dense_output_2 = dense_layer(str_input_layer_2)
                repeated_output_2 = repeat_vector_layer(dense_output_2)  # Repeat the dense output to form a sequence
                lstm_model_2 = lstm_layer(repeated_output_2)
                lstm_model_flat_2 = flat_layer(lstm_model_2)
                input_layers_2.append(str_input_layer_2)
                str_encoded_models_2.append(lstm_model_flat_2)

            else:
                encoder_layer = EncoderLayer(encoding_dim=self.encoding_dim, input_dim=x_train_shapes[idx][0])

                sim_input_layer_1 = Input(shape=x_train_shapes[idx], name=f"Enc_Sim_Input_1_{idx}")
                encoded_model_1 = encoder_layer(sim_input_layer_1)
                input_layers_1.append(sim_input_layer_1)
                sim_encoded_models_1.append(encoded_model_1)

                sim_input_layer_2 = Input(shape=x_train_shapes[idx], name=f"Enc_Sim_Input_2_{idx}")
                encoded_model_2 = encoder_layer(sim_input_layer_2)
                input_layers_2.append(sim_input_layer_2)
                sim_encoded_models_2.append(encoded_model_2)

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

        model = Model(inputs=model_inputs, outputs=main_output, name="GAT_MHA")

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

        # Combine losses
        losses = loss_helper.get_loss_function(self.training_params.loss)

        print("compile_model - loss function:", losses)
        # Compile model with these separate losses and metrics
        model.compile(optimizer='adam', loss=losses, metrics=self.training_params.metrics)

        return model

    def fit_model(self, x_train, y_train, x_val, y_val, x_test, y_test) -> TrainingSummaryDTO:

        x_train_shapes = [x[0].shape for x in x_train]

        train_dataset, val_dataset, test_dataset, train_generator_length, val_generator_length, test_generator_length = (
            self.big_data_loader(x_train, y_train, x_val, y_val, x_test, y_test))

        model = self.build_model(x_train_shapes)

        model = self.compile_model(model)

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

        callbacks = early_stopping

        print('Fit data!')
        history = model.fit(train_dataset, epochs=100, validation_data=val_dataset,
                            steps_per_epoch=train_generator_length, validation_steps=val_generator_length,
                            verbose=1, callbacks=callbacks)

        self.save_plots(history, self.train_id)

        y_pred = model.predict(test_dataset, steps=test_generator_length)

        result = self.calculate_evaluation_metrics(model, x_test, y_test, y_pred=y_pred)

        result.model_info = self.get_model_info(model)

        if self.interaction_data is not None:
            result.data_report = self.get_data_report_split(self.interaction_data, y_train, y_test)

        return result
