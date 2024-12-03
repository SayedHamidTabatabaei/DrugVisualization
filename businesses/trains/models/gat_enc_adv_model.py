from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Concatenate, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import schedules, Adam

from businesses.trains.layers.encoder_layer import EncoderLayer
from businesses.trains.layers.gat_layer import GATLayer
from businesses.trains.layers.reduce_pooling_layer import ReducePoolingLayer
from businesses.trains.models.train_base_model import TrainBaseModel
from common.enums.category import Category
from common.enums.similarity_type import SimilarityType
from common.helpers import loss_helper
from core.models.hyper_params import HyperParams
from core.models.training_params import TrainingParams
from core.repository_models.training_drug_interaction_dto import TrainingDrugInteractionDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO


def f1_score(y_true, y_pred):
    y_true = K.argmax(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)

    tp = K.sum(K.cast(y_true * y_pred, 'float32'))
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float32'))
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float32'))

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class GatEncAdvTrainModel(TrainBaseModel):
    def __init__(self, train_id: int, categories: dict, num_classes: int, interaction_data: list[TrainingDrugInteractionDTO],
                 training_params: TrainingParams, hyper_params: HyperParams):
        super().__init__(train_id, num_classes)
        self.categories = categories
        self.interaction_data = interaction_data
        self.training_params = training_params
        self.encoding_dim = hyper_params.encoding_dim
        self.gat_units = hyper_params.gat_units
        self.num_heads = hyper_params.num_heads
        self.dense_units = hyper_params.dense_units
        self.droprate = hyper_params.droprate
        self.pooling_mode = hyper_params.pooling_mode

        self.training_params.metrics = ['accuracy', Precision(), Recall()]  # ['accuracy', f1_score]

    def build_model(self, data_categories: dict, x_train_shapes, has_interaction_description: bool = False):
        output_models_1 = []
        output_models_2 = []
        input_layers_1 = []
        input_layers_2 = []

        gat_layer = GATLayer(units=self.gat_units, num_heads=self.num_heads)
        reduce_pooling_layer = ReducePoolingLayer(axis=1, pooling_mode=self.pooling_mode)
        for idx, category in data_categories.items():

            if data_categories[idx] == (Category.Substructure, SimilarityType.Original):

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
                # gat_output_1 = reduce_mean_layer(gat_output_1)  # Aggregate over all nodes
                gat_output_1 = reduce_pooling_layer(gat_output_1)  # Aggregate over all nodes
                output_models_1.append(gat_output_1)

                # GAT processing for SMILES
                gat_output_2 = gat_layer((smiles_input_2, adjacency_input_2))
                # gat_output_2 = reduce_mean_layer(gat_output_2)  # Aggregate over all nodes
                gat_output_2 = reduce_pooling_layer(gat_output_2)  # Aggregate over all nodes
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

        x_train_shapes = [x[0].shape for x in x_train]

        model = self.build_model(self.categories, x_train_shapes, bool(self.interaction_data[0].interaction_description))

        if self.training_params.class_weight:
            class_weights = loss_helper.get_class_weights(y_train)
        else:
            class_weights = None

        initial_lr = 1e-4
        lr_schedule = schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=100 * len(y_train),
            alpha=0.0
        )

        optimizer = Adam(learning_rate=lr_schedule)

        model.compile(optimizer=optimizer,
                      loss=loss_helper.get_loss_function(self.training_params.loss, class_weights),
                      metrics=self.training_params.metrics)

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

        history = model.fit(x_train, y_train, epochs=self.training_params.epoch_num, batch_size=128, validation_data=(x_val, y_val), callbacks=early_stopping)

        self.save_plots(history, self.train_id)

        y_pred = model.predict(x_test)

        result = self.calculate_evaluation_metrics(model, x_test, y_test, y_pred=y_pred)

        result.model_info = self.get_model_info(model)

        if self.interaction_data is not None:
            result.data_report = self.get_data_report_split(self.interaction_data, y_train, y_test)

        return result

# https://chatgpt.com/c/674db9c5-7860-800b-ae05-174eaeca1100
