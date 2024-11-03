import numpy as np
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, Reshape, Flatten, MultiHeadAttention, BatchNormalization
# noinspection PyUnresolvedReferences
from tensorflow.keras.utils import Sequence
# noinspection PyUnresolvedReferences
from tensorflow.keras.models import Model
import tensorflow as tf

from businesses.trains.layers.gat_layer import GATLayer
from businesses.trains.layers.reduce_mean_layer import ReduceMeanLayer
from businesses.trains.train_base_service import TrainBaseService
from common.enums.category import Category
from common.enums.train_models import TrainModel
from common.helpers import loss_helper
from core.models.data_params import DataParams
from core.models.training_parameter_models.split_interaction_similarities_training_parameter_model import SplitInteractionSimilaritiesTrainingParameterModel
from core.models.training_params import TrainingParams
from core.repository_models.training_drug_data_dto import TrainingDrugDataDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.GAT_MHA_DNN


class DataGenerator(Sequence):
    def __init__(self, x_data, y_data, batch_size=32):
        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x_data[0]) / self.batch_size))

    def __getitem__(self, index):
        batch_x = [data[index * self.batch_size:(index + 1) * self.batch_size] for data in self.x_data]
        batch_y = self.y_data[index * self.batch_size:(index + 1) * self.batch_size]
        return tuple([np.array(x) for x in batch_x]), batch_y


def get_output_signature(x_train):
    signature = []
    for i, dataset in enumerate(x_train):
        # Get the shape based on the first element in the list (assuming all elements are the same shape within each sub-list)
        first_element_shape = (None,) + tuple(np.array(dataset[0]).shape)
        signature.append(tf.TensorSpec(shape=first_element_shape, dtype=tf.float32))

    return tuple(signature), tf.TensorSpec(shape=(None,), dtype=tf.float32)


class GatMhaDnnTrainService(TrainBaseService):

    def __init__(self, category: TrainModel):
        super().__init__(category)
        self.encoding_dim = 128
        self.gat_units = 64
        self.num_heads = 4
        self.dense_units = [512, 256]
        self.droprate = 0.3

    @staticmethod
    def build_gat_layer(gat_layer, smiles_input_shape):
        smiles_input_1 = Input(shape=smiles_input_shape, name="Drug1_SMILES_Input")
        smiles_input_2 = Input(shape=smiles_input_shape, name="Drug2_SMILES_Input")

        gat_output_1 = gat_layer(smiles_input_1)
        gat_output_1 = ReduceMeanLayer(axis=1)(gat_output_1)

        gat_output_2 = gat_layer(smiles_input_2)
        gat_output_2 = ReduceMeanLayer(axis=1)(gat_output_2)

        return smiles_input_1, gat_output_1, smiles_input_2, gat_output_2

    def build_model(self, data: TrainingDrugDataDTO, x_train):
        smiles_input_1 = None
        smiles_input_2 = None
        gat_output_1 = None
        gat_output_2 = None

        input_layers_str_1 = []
        input_layers_str_2 = []
        input_layers_other_1 = []
        input_layers_other_2 = []
        attention_outputs_1 = []
        attention_outputs_2 = []

        gat_layer = GATLayer(units=self.gat_units, num_heads=self.num_heads)

        for idx, d in enumerate(data.train_values):
            second_idx = idx + len(data.train_values)

            if d.category == Category.Substructure:
                smiles_input_shape = x_train[idx][0].shape
                smiles_input_1, gat_output_1, smiles_input_2, gat_output_2 = self.build_gat_layer(gat_layer, smiles_input_shape)

            elif d.category.data_type == 'str':
                input_layer_str_1 = Input(shape=(x_train[idx][0].shape[-1],), name=f"Str_Input_1_{idx}")
                input_layers_str_1.append(input_layer_str_1)

                input_layer_str_2 = Input(shape=(x_train[second_idx][0].shape[-1],), name=f"Str_Input_2_{second_idx}")
                input_layers_str_2.append(input_layer_str_2)
            else:
                other_input_shape = (len(x_train[idx][0]),)
                other_input_layer_1 = Input(shape=other_input_shape, name=f"Other_Input_1_{idx}")
                input_layers_other_1.append(other_input_layer_1)

                other_input_layer_2 = Input(shape=other_input_shape, name=f"Other_Input_2_{idx}")
                input_layers_other_2.append(other_input_layer_2)

        for i, str_layer in enumerate(input_layers_str_1):
            for j, other_layer in enumerate(input_layers_other_1):
                attention_layer = MultiHeadAttention(num_heads=8, key_dim=64, name=f"MultiHeadAttention_{i}_{j}")

                reshaped_other_input = Reshape((1, input_layers_other_1[j].shape[-1]))(input_layers_other_1[j])
                attention_output = attention_layer(query=input_layers_str_1[i], key=reshaped_other_input, value=reshaped_other_input)
                attention_output = Flatten()(attention_output)
                attention_outputs_1.append(attention_output)

                reshaped_other_input = Reshape((1, input_layers_other_2[j].shape[-1]))(input_layers_other_2[j])
                attention_output = attention_layer(query=input_layers_str_2[i], key=reshaped_other_input, value=reshaped_other_input)
                attention_output = Flatten()(attention_output)
                attention_outputs_2.append(attention_output)

        combined_drug_1 = Concatenate(name="CombinedAttentionOutput_1")([gat_output_1] + attention_outputs_1)
        combined_drug_2 = Concatenate(name="CombinedAttentionOutput_2")([gat_output_2] + attention_outputs_2)

        combined = Concatenate()([combined_drug_1, combined_drug_2])

        train_in = combined
        for units in self.dense_units:
            train_in = Dense(units, activation="relu")(train_in)
            train_in = BatchNormalization()(train_in)
            train_in = Dropout(self.droprate)(train_in)

        output = Dense(self.num_classes, activation="softmax")(train_in)

        model_inputs = [smiles_input_1] + input_layers_other_1 + input_layers_str_1 + [smiles_input_2] + input_layers_other_2 + input_layers_str_2
        model = Model(inputs=model_inputs, outputs=output)

        return model

    def train(self, parameters: SplitInteractionSimilaritiesTrainingParameterModel) -> (TrainingSummaryDTO, object):
        # x_train, x_test, y_train, y_test = super().split_deepface_train_test(parameters.drug_data, parameters.interaction_data, train_id=parameters.train_id,
        #                                                                      mean_of_text_embeddings=False, as_ndarray=False)
        #
        # train_generator = DataGenerator(x_train, y_train, batch_size=32)
        # test_generator = DataGenerator(x_test, y_test, batch_size=32)
        #
        # model = self.build_model(parameters.drug_data[0], train_generator.x_data)
        #
        # output_signature = get_output_signature(x_train)
        #
        # train_dataset = tf.data.Dataset.from_generator(lambda: (batch for batch in train_generator), output_signature=output_signature)
        # test_dataset = tf.data.Dataset.from_generator(lambda: (batch for batch in test_generator), output_signature=output_signature)
        #
        # model.compile(optimizer='adam', loss=loss_helper.get_loss_function(parameters.loss_function), metrics=['accuracy'])
        #
        # print('Fit data!')
        # history = model.fit(train_dataset, epochs=50, validation_data=test_dataset)
        #
        # result = self.calculate_evaluation_metrics(model, x_test, y_test)
        #
        # self.plot_accuracy(history, parameters.train_id)
        # self.plot_loss(history, parameters.train_id)
        #
        # result.model_info = self.get_model_info(model)
        #
        # if parameters.interaction_data is not None:
        #     result.data_report = self.get_data_report_split(parameters.interaction_data, y_train, y_test)
        #
        # return result

        x_train, x_test, y_train, y_test = super().split_deepface_train_test(parameters.drug_data, parameters.interaction_data, train_id=parameters.train_id,
                                                                             mean_of_text_embeddings=False)

        model = self.build_model(parameters.drug_data[0], x_train)

        return super().fit_dnn_model(data_params=DataParams(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test),
                                     training_params=TrainingParams(train_id=parameters.train_id, optimizer='adam', loss=parameters.loss_function,
                                                                    class_weight=parameters.class_weight),
                                     model=model,
                                     interactions=parameters.interaction_data)
