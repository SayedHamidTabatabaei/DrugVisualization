import contextlib
import io
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_score, \
    recall_score, log_loss
from sklearn.preprocessing import label_binarize
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping

from businesses.trains.layers.data_generator import DataGenerator
from common.enums.training_result_type import TrainingResultType
from common.helpers import loss_helper
from core.models.training_params import TrainingParams
from core.repository_models.training_drug_interaction_dto import TrainingDrugInteractionDTO
from core.repository_models.training_result_detail_summary_dto import TrainingResultDetailSummaryDTO
from core.repository_models.training_result_summary_dto import TrainingResultSummaryDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO


class TrainBaseModel:
    def __init__(self, train_id: int, num_classes: int):
        self.train_id = train_id
        self.num_classes = num_classes

    def fit_train(self, x_train, y_train, x_val, y_val, x_test, y_test):
        pass

    def base_fit_model(self, model, training_params: TrainingParams, interaction_data: list[TrainingDrugInteractionDTO],
                       x_train, y_train, x_val, y_val, x_test, y_test) -> TrainingSummaryDTO:

        if training_params.class_weight:
            class_weights = loss_helper.get_class_weights(y_train)
        else:
            class_weights = None

        model.compile(optimizer=training_params.optimizer,
                      loss=loss_helper.get_loss_function(training_params.loss, class_weights),
                      metrics=training_params.metrics)

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

        history = model.fit(x_train, y_train, epochs=training_params.epoch_num, batch_size=128, validation_data=(x_val, y_val), callbacks=early_stopping)

        self.save_plots(history, self.train_id)

        y_pred = model.predict(x_test)

        result = self.calculate_evaluation_metrics(model, x_test, y_test, y_pred=y_pred)

        result.model_info = self.get_model_info(model)

        if interaction_data is not None:
            result.data_report = self.get_data_report_split(interaction_data, y_train, y_test)

        return result

    @staticmethod
    def get_image_folder_name(train_id: int) -> str:
        folder_name = f"training_plots/{train_id}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        return folder_name

    def save_plots(self, history, train_id: int, number: int = None):

        folder_name = self.get_image_folder_name(train_id)
        os.makedirs(folder_name, exist_ok=True)

        def design_save_plot(plot_name):
            plt.figure(figsize=(12, 4))

            plot_title = plot_name.replace('_', ' ')

            plt.plot(history.history[plot_name], label=f'Training {plot_title}')
            plt.plot(history.history[f'val_{plot_name}'], label=f'Validation {plot_title}')
            plt.title(f'{plot_title} over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel(f'{plot_title}')
            plt.legend()

            plot_name = f'{plot_name}_plot_{number}.png' if number else f'{plot_name}_plot.png'

            plot_path = os.path.join(folder_name, f'{plot_name}')
            plt.savefig(plot_path)

            plt.close()

        [design_save_plot(k) for k in history.history.keys() if not k.startswith('val_')]

    def calculate_evaluation_metrics(self, model, x_test, y_test, y_pred=None, is_labels_categorical: bool = False) -> TrainingSummaryDTO:

        training_results: list[TrainingResultSummaryDTO] = []

        if y_pred is None:
            y_pred = model.predict(x_test)

        y_pred_classes = np.argmax(y_pred, axis=1) if not is_labels_categorical else y_pred
        y_test_classes = np.argmax(y_test, axis=1) if not is_labels_categorical else y_test

        y_pred = y_pred if not is_labels_categorical else np.eye(self.num_classes)[y_pred_classes]
        y_test = y_test if not is_labels_categorical else np.eye(self.num_classes)[y_test_classes]

        incorrect_indices = np.where(y_pred_classes != y_test_classes)[0]
        incorrect_predictions = {int(idx): int(y_pred_classes[idx]) for idx in incorrect_indices}

        accuracy = accuracy_score(y_test_classes, y_pred_classes)
        training_results.append(TrainingResultSummaryDTO(TrainingResultType.accuracy, accuracy))

        print('Calculate loss!')
        # loss, total_accuracy = model.evaluate(x_test, y_test)
        loss = log_loss(y_test, y_pred)
        training_results.append(TrainingResultSummaryDTO(TrainingResultType.loss, loss))

        f1_score_weighted = f1_score(y_test_classes, y_pred_classes, average='weighted', zero_division=0)
        training_results.append(TrainingResultSummaryDTO(TrainingResultType.f1_score_weighted, f1_score_weighted))

        f1_score_micor = f1_score(y_test_classes, y_pred_classes, average='micro', zero_division=0)
        training_results.append(TrainingResultSummaryDTO(TrainingResultType.f1_score_micro, f1_score_micor))

        f1_score_macro = f1_score(y_test_classes, y_pred_classes, average='macro', zero_division=0)
        training_results.append(TrainingResultSummaryDTO(TrainingResultType.f1_score_macro, f1_score_macro))

        # For multi-class AUC and AUPR, you need to binarize the labels
        y_test_bin = label_binarize(y_test_classes, classes=range(self.num_classes))
        list(set(np.argmax(y_test_bin, axis=1)))

        auc_weighted = roc_auc_score(y_test_bin, y_pred, average='weighted', multi_class='ovr')
        training_results.append(TrainingResultSummaryDTO(TrainingResultType.auc_weighted, auc_weighted))

        auc_micro = roc_auc_score(y_test_bin, y_pred, average='micro', multi_class='ovr')
        training_results.append(TrainingResultSummaryDTO(TrainingResultType.auc_micro, auc_micro))

        auc_macro = roc_auc_score(y_test_bin, y_pred, average='macro', multi_class='ovr')
        training_results.append(TrainingResultSummaryDTO(TrainingResultType.auc_macro, auc_macro))

        aupr_weighted = average_precision_score(y_test_bin, y_pred, average='weighted')
        training_results.append(TrainingResultSummaryDTO(TrainingResultType.aupr_weighted, aupr_weighted))

        aupr_micro = average_precision_score(y_test_bin, y_pred, average='micro')
        training_results.append(TrainingResultSummaryDTO(TrainingResultType.aupr_micro, aupr_micro))

        aupr_macro = average_precision_score(y_test_bin, y_pred, average='macro')
        training_results.append(TrainingResultSummaryDTO(TrainingResultType.aupr_macro, aupr_macro))

        precision_weighted = precision_score(y_test_classes, y_pred_classes, average='weighted', zero_division=0)
        training_results.append(TrainingResultSummaryDTO(TrainingResultType.precision_weighted, precision_weighted))

        precision_micro = precision_score(y_test_classes, y_pred_classes, average='micro', zero_division=0)
        training_results.append(TrainingResultSummaryDTO(TrainingResultType.precision_micro, precision_micro))

        precision_macro = precision_score(y_test_classes, y_pred_classes, average='macro', zero_division=0)
        training_results.append(TrainingResultSummaryDTO(TrainingResultType.precision_macro, precision_macro))

        recall_weighted = recall_score(y_test_classes, y_pred_classes, average='weighted', zero_division=0)
        training_results.append(TrainingResultSummaryDTO(TrainingResultType.recall_weighted, recall_weighted))

        recall_micro = recall_score(y_test_classes, y_pred_classes, average='micro', zero_division=0)
        training_results.append(TrainingResultSummaryDTO(TrainingResultType.recall_micro, recall_micro))

        recall_macro = recall_score(y_test_classes, y_pred_classes, average='macro', zero_division=0)
        training_results.append(TrainingResultSummaryDTO(TrainingResultType.recall_macro, recall_macro))

        # Binarize the labels for AUC and AUPR calculation
        y_test_bin = label_binarize(y_test_classes, classes=range(self.num_classes))
        results_per_labels: list[TrainingResultDetailSummaryDTO] = []

        print('Calculate classes Evaluations!')

        precision_per_class = precision_score(y_test_classes, y_pred_classes, average=None, zero_division=0)
        recall_per_class = recall_score(y_test_classes, y_pred_classes, average=None, zero_division=0)

        for i in range(self.num_classes):
            class_accuracy = accuracy_score(y_test_classes == i, y_pred_classes == i)

            class_f1 = f1_score(y_test_classes == i, y_pred_classes == i, average='binary')

            class_auc = roc_auc_score(y_test_bin[:, i], y_pred[:, i])

            class_aupr = average_precision_score(y_test_bin[:, i], y_pred[:, i])

            results_per_labels.append(TrainingResultDetailSummaryDTO(training_label=i,
                                                                     f1_score=class_f1,
                                                                     accuracy=class_accuracy,
                                                                     auc=class_auc,
                                                                     aupr=class_aupr,
                                                                     recall=recall_per_class[i],
                                                                     precision=precision_per_class[i]))

        return TrainingSummaryDTO(training_results=training_results,
                                  model=model,
                                  training_result_details=results_per_labels,
                                  data_report=None,
                                  model_info=None,
                                  fold_result_details=None,
                                  incorrect_predictions=incorrect_predictions)

    @staticmethod
    def get_model_info(model):

        summary_str = io.StringIO()
        with contextlib.redirect_stdout(summary_str):
            model.summary()
        model_summary = summary_str.getvalue()

        # Create dictionary to save as JSON
        model_info = {
            "model_summary": model_summary
        }

        return model_info

    @staticmethod
    def get_data_report(interactions: list[TrainingDrugInteractionDTO]):
        df = pd.DataFrame([vars(interaction) for interaction in interactions])

        grouped_counts = df.groupby('interaction_type').size()

        grouped_counts_dict = grouped_counts.reset_index().to_dict(orient='records')

        return [{g['interaction_type']: g[0]} for g in grouped_counts_dict]

    def get_data_report_by_labels(self, labels: list[int], is_labels_categorical: bool = False):
        indices = np.argmax(labels, axis=1) if not is_labels_categorical else labels

        counts = np.bincount(indices, minlength=self.num_classes)

        return [{i: c} for i, c in enumerate(counts)]

    def get_data_report_split(self, interactions: list[TrainingDrugInteractionDTO], y_train, y_test,
                              is_labels_categorical: bool = False):
        return {
            "total_count": len(interactions),
            "train_count": len(y_train),
            "test_count": len(y_test),
            "total_report": self.get_data_report(interactions),
            "train_report": self.get_data_report_by_labels(y_train, is_labels_categorical),
            "test_report": self.get_data_report_by_labels(y_test, is_labels_categorical)
        }

    @staticmethod
    def calculate_shapes(x_train):
        x_train_shapes = [x.shape for x in x_train]

        return x_train_shapes

    @staticmethod
    def get_output_signature(x_train, y_train, multiple_output:bool = False):
        x_signature = []
        for dataset in x_train:
            # Get the shape based on the first element in the list (assuming all elements are the same shape within each sub-list)
            first_element_shape = (None,) + tuple(np.array(dataset[0]).shape)
            x_signature.append(tf.TensorSpec(shape=first_element_shape, dtype=dataset[0].dtype))

        if not multiple_output:

            y_signature = tf.TensorSpec(shape=(None, len(y_train[0])), dtype=y_train[0].dtype)  # Batch size, num_classes
            return tuple(x_signature), y_signature
        else:
            y_signature = []
            for dataset in y_train:
                # Get the shape based on the first element in the list (assuming all elements are the same shape within each sub-list)
                first_element_shape = (None,) + tuple(np.array(dataset[0]).shape)
                y_signature.append(tf.TensorSpec(shape=first_element_shape, dtype=dataset[0].dtype))
            return tuple(x_signature), tuple(y_signature)

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

    def big_data_loader(self, x_train, y_train, x_val, y_val, x_test, y_test, multiple_output:bool = False):

        y_train = self.generate_y_data_autoencoder(x_train, y_train)
        y_val = self.generate_y_data_autoencoder(x_val, y_val)
        y_test = self.generate_y_data_autoencoder(x_test, y_test)

        train_generator = DataGenerator(x_train, y_train, batch_size=256, drop_remainder=True, multiple_output=multiple_output)
        val_generator = DataGenerator(x_val, y_val, batch_size=256, drop_remainder=True, multiple_output=multiple_output)
        test_generator = DataGenerator(x_test, y_test, batch_size=1, drop_remainder=True, multiple_output=multiple_output)

        output_signature = self.get_output_signature(x_train, y_train, multiple_output)

        train_dataset = tf.data.Dataset.from_generator(lambda: iter(train_generator), output_signature=output_signature).repeat()
        val_dataset = tf.data.Dataset.from_generator(lambda: iter(val_generator), output_signature=output_signature).repeat()
        test_dataset = tf.data.Dataset.from_generator(lambda: iter(test_generator), output_signature=output_signature).repeat()

        return train_dataset, val_dataset, test_dataset, len(train_generator), len(val_generator), len(test_generator)

    def generate_y_data_autoencoder(self, x_data, y_data):
        return y_data
