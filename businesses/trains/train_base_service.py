import contextlib
import io
import json
import math
import os
import random
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_score, \
    recall_score, log_loss
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import label_binarize
# noinspection PyUnresolvedReferences
from tensorflow.keras.callbacks import EarlyStopping
# noinspection PyUnresolvedReferences
from tensorflow.keras.preprocessing.sequence import pad_sequences
# noinspection PyUnresolvedReferences
from tensorflow.keras.preprocessing.text import Tokenizer
# noinspection PyUnresolvedReferences
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from common.enums.category import Category
from common.enums.train_models import TrainModel
from common.enums.training_result_type import TrainingResultType
from common.helpers import loss_helper
from core.models.data_params import DataParams
from core.models.training_parameter_base_model import TrainingParameterBaseModel
from core.models.training_params import TrainingParams
from core.repository_models.training_drug_data_dto import TrainingDrugDataDTO
from core.repository_models.training_drug_interaction_dto import TrainingDrugInteractionDTO
from core.repository_models.training_drug_train_values_dto import TrainingDrugTrainValuesDTO
from core.repository_models.training_result_detail_summary_dto import TrainingResultDetailSummaryDTO
from core.repository_models.training_result_summary_dto import TrainingResultSummaryDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO


class TrainBaseService:
    category: TrainModel

    def __init__(self, category: TrainModel):
        self.category = category
        self.num_classes = 65
        self.num_folds = 5
        self.usage_model_file = 'usage_models'
        self.pca_model_file_name = 'PCA_tanimoto_model_50.pkl'

    def save_image(self, train_id):
        pass

    def train(self, parameters: TrainingParameterBaseModel) -> TrainingSummaryDTO:
        pass

    def calculate_evaluation_metrics_multi_model(self, model, x_test, y_test, output_name="main_output", is_labels_categorical: bool = False) \
            -> TrainingSummaryDTO:

        y_pred = model.predict(x_test)[output_name]

        return self.calculate_evaluation_metrics(model, x_test, y_test, y_pred, is_labels_categorical)

    def calculate_evaluation_metrics(self, model, x_test, y_test, y_pred=None, is_labels_categorical: bool = False) -> TrainingSummaryDTO:

        training_results: list[TrainingResultSummaryDTO] = []

        if y_pred is None:
            y_pred = model.predict(x_test)

        y_pred_classes = np.argmax(y_pred, axis=1) if not is_labels_categorical else y_pred
        y_test_classes = np.argmax(y_test, axis=1) if not is_labels_categorical else y_test

        y_pred = y_pred if not is_labels_categorical else np.eye(self.num_classes)[y_pred_classes]
        y_test = y_test if not is_labels_categorical else np.eye(self.num_classes)[y_test_classes]

        accuracy = accuracy_score(y_test_classes, y_pred_classes)
        training_results.append(TrainingResultSummaryDTO(TrainingResultType.accuracy, accuracy))

        print('Calculate loss!')
        # loss, total_accuracy = model.evaluate(x_test, y_test)
        loss = log_loss(y_test, y_pred)
        training_results.append(TrainingResultSummaryDTO(TrainingResultType.loss, loss))

        f1_score_weighted = f1_score(y_test_classes, y_pred_classes, average='weighted')
        training_results.append(TrainingResultSummaryDTO(TrainingResultType.f1_score_weighted, f1_score_weighted))

        f1_score_micor = f1_score(y_test_classes, y_pred_classes, average='micro')
        training_results.append(TrainingResultSummaryDTO(TrainingResultType.f1_score_micro, f1_score_micor))

        f1_score_macro = f1_score(y_test_classes, y_pred_classes, average='macro')
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

        precision_weighted = precision_score(y_test_classes, y_pred_classes, average='weighted')
        training_results.append(TrainingResultSummaryDTO(TrainingResultType.precision_weighted, precision_weighted))

        precision_micro = precision_score(y_test_classes, y_pred_classes, average='micro')
        training_results.append(TrainingResultSummaryDTO(TrainingResultType.precision_micro, precision_micro))

        precision_macro = precision_score(y_test_classes, y_pred_classes, average='macro')
        training_results.append(TrainingResultSummaryDTO(TrainingResultType.precision_macro, precision_macro))

        recall_weighted = recall_score(y_test_classes, y_pred_classes, average='weighted')
        training_results.append(TrainingResultSummaryDTO(TrainingResultType.recall_weighted, recall_weighted))

        recall_micro = recall_score(y_test_classes, y_pred_classes, average='micro')
        training_results.append(TrainingResultSummaryDTO(TrainingResultType.recall_micro, recall_micro))

        recall_macro = recall_score(y_test_classes, y_pred_classes, average='macro')
        training_results.append(TrainingResultSummaryDTO(TrainingResultType.recall_macro, recall_macro))

        # Binarize the labels for AUC and AUPR calculation
        y_test_bin = label_binarize(y_test_classes, classes=range(self.num_classes))
        results_per_labels: list[TrainingResultDetailSummaryDTO] = []

        print('Calculate classes Evaluations!')

        precision_per_class = precision_score(y_test_classes, y_pred_classes, average=None)
        recall_per_class = recall_score(y_test_classes, y_pred_classes, average=None)

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
                                  fold_result_details=None)

    @staticmethod
    def plot_accuracy(history, train_id):

        folder_name = TrainBaseService.get_image_folder_name(train_id)

        plt.figure(figsize=(12, 4))

        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plot_path = os.path.join(folder_name, 'accuracy_plot.png')
        plt.savefig(plot_path)

        plt.close()

    @staticmethod
    def plot_loss(history, train_id):

        folder_name = TrainBaseService.get_image_folder_name(train_id)

        plt.figure(figsize=(12, 4))

        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plot_path = os.path.join(folder_name, 'loss_plot.png')
        plt.savefig(plot_path)

        plt.close()

    @staticmethod
    def get_image_folder_name(train_id: int) -> str:
        folder_name = f"training_plots/{train_id}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        return folder_name

    @staticmethod
    def plot_radial(values):

        num_categories = len(values)

        angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()

        values = np.concatenate((values, [values[0]]))
        angles += [angles[0]]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        ax.plot(angles, values, linewidth=2, linestyle='solid')

        labels = [i for i in range(num_categories)]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8)

        ax.set_ylim(0, 1)

        return plt

    @staticmethod
    def plot_accuracy_radial(values, train_id: int):

        folder_name = TrainBaseService.get_image_folder_name(train_id)

        plt_radial = TrainBaseService.plot_radial(values)

        plt_radial.title('Accuracy')

        plot_path = os.path.join(folder_name, 'accuracies.png')
        plt_radial.savefig(plot_path)

        plt_radial.close()

    @staticmethod
    def plot_f1_score_radial(values, train_id: int):

        folder_name = TrainBaseService.get_image_folder_name(train_id)

        plt_radial = TrainBaseService.plot_radial(values)

        plt_radial.title('F1 Score')

        plot_path = os.path.join(folder_name, 'f1_score.png')
        plt_radial.savefig(plot_path)

        plt_radial.close()

    @staticmethod
    def plot_auc_radial(values, train_id: int):

        folder_name = TrainBaseService.get_image_folder_name(train_id)

        plt_radial = TrainBaseService.plot_radial(values)

        plt_radial.title('AUC')

        plot_path = os.path.join(folder_name, 'auc.png')
        plt_radial.savefig(plot_path)

        plt_radial.close()

    # @tf.function
    @staticmethod
    def create_input_tensors_ragged(x_train, x_test):
        start_time = time.time()
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

        x_train_ragged = [tf.ragged.constant(d) for d in tqdm(x_train, desc="Creating Ragged Train data")]
        x_test_ragged = [tf.ragged.constant(d) for d in tqdm(x_test, desc="Creating Ragged Test data")]

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Finished time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Execution time: {execution_time} seconds")

        return x_train_ragged, x_test_ragged

    @staticmethod
    def create_input_tensors_pad(x_train, x_test):

        max_len_train = max(max(len(seq) for seq in x_train[i]) for i in range(len(x_train)))
        max_len_test = max(max(len(seq) for seq in x_test[i]) for i in range(len(x_test)))
        max_len = max(max_len_train, max_len_test)

        x_train_padded = []
        for i in tqdm(range(len(x_train)), "Padding train data ..."):
            padded = TrainBaseService.process_padding_data(max_len, x_train[i])
            x_train_padded.append(padded)

        x_test_padded = []
        for i in tqdm(range(len(x_test)), "Padding test data ..."):
            padded = TrainBaseService.process_padding_data(max_len, x_test[i])
            x_test_padded.append(padded)

        return x_train_padded, x_test_padded

    @staticmethod
    def create_input_tensors_pad_with_val(x_train, x_val, x_test):

        max_len_train = max(max(len(seq) for seq in x_train[i]) for i in range(len(x_train)))
        max_len_val = max(max(len(seq) for seq in x_val[i]) for i in range(len(x_val)))
        max_len_test = max(max(len(seq) for seq in x_test[i]) for i in range(len(x_test)))
        max_len = max(max_len_train, max_len_val, max_len_test)

        x_train_padded = []
        for i in tqdm(range(len(x_train)), "Padding train data ..."):
            padded = TrainBaseService.process_padding_data(max_len, x_train[i])
            x_train_padded.append(padded)

        x_val_padded = []
        for i in tqdm(range(len(x_val)), "Padding val data ..."):
            padded = TrainBaseService.process_padding_data(max_len, x_val[i])
            x_val_padded.append(padded)

        x_test_padded = []
        for i in tqdm(range(len(x_test)), "Padding test data ..."):
            padded = TrainBaseService.process_padding_data(max_len, x_test[i])
            x_test_padded.append(padded)

        return x_train_padded, x_val_padded, x_test_padded

    @staticmethod
    def process_padding_data(max_len, x_data):
        # Check if the data is numeric or SMILES
        if isinstance(x_data[0][0], (int, float, np.number)):
            # If numeric, pad directly
            padded = np.array(x_data) if all(len(seq) == max_len for seq in x_data) else pad_sequences(x_data, maxlen=max_len, padding='post')
        else:
            tokenizer = Tokenizer(char_level=True)  # Character-level tokenizer
            all_smiles = [str(smile) for sublist in x_data for smile in sublist]  # Flatten list
            tokenizer.fit_on_texts(all_smiles)

            smiles_list = x_data[55]  # 55 was i
            smiles_sequences = [tokenizer.texts_to_sequences([str(smile)])[0] for smile in smiles_list]
            padded = pad_sequences(smiles_sequences, maxlen=max_len, padding='post')
        return padded

    @staticmethod
    def create_input_tensors_flat(x_train, x_test):
        x_train_stacked = np.concatenate(x_train, axis=1)  # Stack the arrays along the feature axis
        x_test_stacked = np.concatenate(x_test, axis=1)

        x_train_flat = x_train_stacked.reshape(x_train_stacked.shape[0], -1)
        x_test_flat = x_test_stacked.reshape(x_test_stacked.shape[0], -1)

        return x_train_flat, x_test_flat

    @staticmethod
    def create_input_tensors_flat_with_validation(x_train, x_val, x_test):
        x_train_stacked = np.concatenate(x_train, axis=1)  # Stack the arrays along the feature axis
        x_val_stacked = np.concatenate(x_val, axis=1)
        x_test_stacked = np.concatenate(x_test, axis=1)

        x_train_flat = x_train_stacked.reshape(x_train_stacked.shape[0], -1)
        x_val_flat = x_val_stacked.reshape(x_val_stacked.shape[0], -1)
        x_test_flat = x_test_stacked.reshape(x_test_stacked.shape[0], -1)

        return x_train_flat, x_val_flat, x_test_flat

    @staticmethod
    def pad_sequences(data, maxlen=None, padding_value='0'):
        return tf.keras.preprocessing.sequence.pad_sequences(data, maxlen=maxlen, padding='post', value=padding_value)

    @staticmethod
    def unique_category(train_values: list[TrainingDrugTrainValuesDTO]):
        unique_categories = {}
        index = 0

        for c in train_values:
            if c.category not in unique_categories.values():
                unique_categories[index] = c.category
                if c.category == Category.Substructure:
                    index += 1
                index += 1

        return unique_categories

    def fit_dnn_model(self, data_params: DataParams, training_params: TrainingParams, model, interactions: (list[TrainingDrugInteractionDTO] | None) = None) \
            -> TrainingSummaryDTO:
        if training_params.class_weight:
            class_weights = loss_helper.get_class_weights(data_params.y_train)
        else:
            class_weights = None

        model.compile(optimizer=training_params.optimizer,
                      loss=loss_helper.get_loss_function(training_params.loss, class_weights),
                      metrics=training_params.metrics)

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

        print('Fit data!')
        history = model.fit(data_params.x_train, data_params.y_train,
                            epochs=100, batch_size=256,
                            validation_data=(data_params.x_val, data_params.y_val),
                            class_weight=class_weights, callbacks=early_stopping)

        result = self.calculate_evaluation_metrics(model, data_params.x_test, data_params.y_test)

        self.plot_accuracy(history, training_params.train_id)
        self.plot_loss(history, training_params.train_id)

        result.model_info = self.get_model_info(model)

        if interactions is not None:
            result.data_report = self.get_data_report_split(interactions, data_params.y_train, data_params.y_test)

        return result

    def fit_dnn_model_multi_model(self, data_params: DataParams, training_params: TrainingParams, multi_task_model,
                                  interactions: (list[TrainingDrugInteractionDTO] | None) = None) \
            -> TrainingSummaryDTO:

        if training_params.class_weight:
            class_weights = loss_helper.get_class_weights(data_params.y_train)
        else:
            class_weights = None

        multi_task_model.compile(optimizer=training_params.optimizer,
                                 loss={'reconstruction_output': 'sigmoid', 'main_output': loss_helper.get_loss_function(training_params.loss, class_weights)},
                                 loss_weights={'reconstruction_output': 1.0, 'main_output': 0.5},
                                 metrics={'main_output': training_params.metrics})

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

        print('Fit data!')
        history = multi_task_model.fit(
            x=data_params.x_train,
            y={'reconstruction_output': data_params.x_train, 'main_output': data_params.y_train},
            epochs=100,
            batch_size=256,
            validation_data=(
                data_params.x_val,
                {'reconstruction_output': data_params.x_val, 'main_output': data_params.y_val}
            ),
            class_weight=class_weights,
            callbacks=[early_stopping]
        )

        result = self.calculate_evaluation_metrics_multi_model(multi_task_model, data_params.x_test, data_params.y_test)

        self.plot_accuracy(history, training_params.train_id)
        self.plot_loss(history, training_params.train_id)

        result.model_info = self.get_model_info(multi_task_model)

        if interactions is not None:
            result.data_report = self.get_data_report_split(interactions, data_params.y_train, data_params.y_test)

        return result

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

    def get_data_report_split(self, interactions: list[TrainingDrugInteractionDTO], y_train, y_test, is_labels_categorical: bool = False):
        return {
            "total_count": len(interactions),
            "train_count": len(y_train),
            "test_count": len(y_test),
            "total_report": self.get_data_report(interactions),
            "train_report": self.get_data_report_by_labels(y_train, is_labels_categorical),
            "test_report": self.get_data_report_by_labels(y_test, is_labels_categorical)
        }

    @staticmethod
    def calculate_fold_results(fold_results: list[TrainingSummaryDTO]):

        results = []
        mean_result: TrainingSummaryDTO

        for idx, result in enumerate(fold_results):
            result = {
                "k": idx + 1,
                "training_results": {
                    r.training_result_type.name: r.result_value for r in result.training_results
                },
                "training_result_details": [
                    {
                        "training_label": r.training_label,
                        "accuracy": r.accuracy,
                        "f1_score": r.f1_score,
                        "auc": r.auc,
                        "aupr": r.aupr,
                        "recall": r.recall,
                        "precision": r.precision
                    }
                    for r in result.training_result_details
                ]
            }

            results.append(result)

        return TrainingSummaryDTO(
            training_results=[
                TrainingResultSummaryDTO(
                    training_result_type=training_result_type,
                    result_value=np.mean([r.result_value for result in fold_results for r in result.training_results if r.training_result_type ==
                                          training_result_type])
                )
                for training_result_type in {r.training_result_type for result in fold_results for r in result.training_results}
            ],
            model=[r.model for r in fold_results],
            data_report=[r.data_report for r in fold_results],
            model_info=[r.model_info for r in fold_results],
            fold_result_details=results,
            training_result_details=[
                TrainingResultDetailSummaryDTO(
                    training_label=training_label,
                    accuracy=np.mean([r.accuracy for result in fold_results for r in result.training_result_details if r.training_label == training_label]),
                    f1_score=np.mean([r.f1_score for result in fold_results for r in result.training_result_details if r.training_label == training_label]),
                    auc=np.mean([r.auc for result in fold_results for r in result.training_result_details if r.training_label == training_label]),
                    aupr=np.mean([r.aupr for result in fold_results for r in result.training_result_details if r.training_label == training_label]),
                    recall=np.mean([r.recall for result in fold_results for r in result.training_result_details if r.training_label == training_label]),
                    precision=np.mean([r.precision for result in fold_results for r in result.training_result_details if r.training_label == training_label])
                )
                for training_label in {r.training_label for result in fold_results for r in result.training_result_details}
            ]
        )

    # region split data
    def manual_k_fold_train_test_data(self, drug_data: list[TrainingDrugDataDTO], interaction_data: list[TrainingDrugInteractionDTO],
                                      categorical_labels: bool = True, padding: bool = False, flat: bool = False,
                                      pca_generating: bool = False, pca_components: int = None,
                                      is_deep_face: bool = False, is_cnn: bool = False,
                                      compare_train_test: bool = True):

        self.num_classes = len(set(item.interaction_type for item in interaction_data))

        all_drugs = [drug.drug_id for drug in drug_data]
        # min_test_drugs_in_folds = math.floor(len(drug_data) / self.num_folds)
        previous_fold_tests = []
        interaction_drugs: dict = self.get_drug_interaction_dict(interaction_data)
        min_test_drugs_in_folds_per_interaction_type = \
            {key: math.floor(len(value) / self.num_folds) if len(value) < 50 else math.ceil(len(value) / self.num_folds)
             for key, value in sorted(interaction_drugs.items(), reverse=True)}

        for k in range(1, self.num_folds + 1):
            active_drugs_on_test = {
                key: {drug for drug in drugs if drug not in previous_fold_tests}
                for key, drugs in sorted(interaction_drugs.items(), reverse=True)
            }

            test_drug_ids: list[int] = []
            train_drug_ids: list[int] = []

            if k == self.num_folds:
                test_drug_ids = [drug for drugs in active_drugs_on_test.values() for drug in drugs]
                train_drug_ids = [drug for drug in all_drugs if drug not in test_drug_ids]

            else:
                for key, drugs in active_drugs_on_test.items():

                    temp_test = list(drugs.intersection(test_drug_ids))

                    needed_data_count = min_test_drugs_in_folds_per_interaction_type[key] - len(temp_test)

                    if 0 < needed_data_count < len([d for d in drugs if d not in test_drug_ids and d not in train_drug_ids]):
                        temp_test = temp_test + random.sample([d for d in drugs if d not in test_drug_ids and d not in train_drug_ids], needed_data_count)

                    if key == 0:
                        pass

                    test_drug_ids = list(set(test_drug_ids + temp_test))

                    temp_train = [drug for drug in interaction_drugs[key] if drug not in temp_test]
                    train_drug_ids = train_drug_ids + temp_train

            test_drug_ids = list(set(test_drug_ids))
            train_drug_ids = list(set(train_drug_ids))

            previous_fold_tests.extend(test_drug_ids)

            # Filter interaction_data based on the defined conditions
            train_interactions = [
                interaction for interaction in interaction_data
                if interaction.drug_1 in train_drug_ids and interaction.drug_2 in train_drug_ids
            ]

            if compare_train_test:
                test_interactions = [
                    interaction for interaction in interaction_data
                    if interaction.drug_1 in test_drug_ids or interaction.drug_2 in test_drug_ids
                ]
            else:
                test_interactions = [
                    interaction for interaction in interaction_data
                    if interaction.drug_1 in test_drug_ids and interaction.drug_2 in test_drug_ids
                ]

            y_train = [i.interaction_type for i in train_interactions]
            y_test = [i.interaction_type for i in test_interactions]

            if is_deep_face:
                x_train = self.generate_deepface_x_values(drug_data, train_interactions, train_drug_ids, pca_generating=pca_generating)
                x_test = self.generate_deepface_x_values(drug_data, test_interactions, train_drug_ids, pca_generating=pca_generating)
            elif is_cnn:
                x_train = self.generate_cnn_x_values(drug_data, train_interactions, train_drug_ids)
                x_test = self.generate_cnn_x_values(drug_data, test_interactions, train_drug_ids)
            else:
                x_train = self.generate_x_values(drug_data, train_interactions, train_drug_ids, pca_generating=pca_generating, pca_components=pca_components)
                x_test = self.generate_x_values(drug_data, test_interactions, train_drug_ids, pca_generating=pca_generating, pca_components=pca_components)

            if categorical_labels:
                y_train = to_categorical(y_train, num_classes=self.num_classes)
                y_test = to_categorical(y_test, num_classes=self.num_classes)

            if padding:
                x_train, x_test = self.create_input_tensors_pad(x_train, x_test)

            if flat:
                x_train, x_test = self.create_input_tensors_flat(x_train, x_test)

            yield [np.array(x) for x in x_train], [np.array(x) for x in x_test], y_train, y_test

    @staticmethod
    def get_drug_interaction_dict(interaction_data):
        interaction_dict = defaultdict(set)
        [interaction_dict[interaction.interaction_type].update([interaction.drug_1, interaction.drug_2]) for interaction in interaction_data]
        return dict(sorted(interaction_dict.items(), reverse=True))

    def split_train_test(self, drug_data: list[TrainingDrugDataDTO], interaction_data: list[TrainingDrugInteractionDTO], train_id: int,
                         categorical_labels: bool = True, padding: bool = False, flat: bool = False, mean_of_text_embeddings: bool = True,
                         pca_generating: bool = False, pca_components: int = None):

        self.num_classes = len(set(item.interaction_type for item in interaction_data))

        drug_ids = [drug.drug_id for drug in drug_data]

        x = self.generate_x_values(drug_data, interaction_data, drug_ids, mean_of_text_embeddings=mean_of_text_embeddings, pca_generating=pca_generating,
                                   pca_components=pca_components)
        y = np.array([item.interaction_type for item in interaction_data])  # Extract labels

        x_train, x_test, y_train, y_test = self.split_on_interactions(x, y, interactions=interaction_data, train_id=train_id)

        if categorical_labels:
            y_train = to_categorical(y_train, num_classes=self.num_classes)
            y_test = to_categorical(y_test, num_classes=self.num_classes)

        if padding:
            x_train, x_test = self.create_input_tensors_pad(x_train, x_test)

        if flat:
            x_train, x_test = self.create_input_tensors_flat(x_train, x_test)

        return x_train, x_test, y_train, y_test

    def split_train_val_test(self, drug_data: list[TrainingDrugDataDTO], interaction_data: list[TrainingDrugInteractionDTO], train_id: int,
                             categorical_labels: bool = True, padding: bool = False, flat: bool = False, mean_of_text_embeddings: bool = True,
                             pca_generating: bool = False, pca_components: int = None):

        self.num_classes = len(set(item.interaction_type for item in interaction_data))

        drug_ids = [drug.drug_id for drug in drug_data]

        x = self.generate_x_values(drug_data, interaction_data, drug_ids, mean_of_text_embeddings=mean_of_text_embeddings, pca_generating=pca_generating,
                                   pca_components=pca_components)
        y = np.array([item.interaction_type for item in interaction_data])  # Extract labels

        x_train, x_val, x_test, y_train, y_val, y_test = self.split_on_interactions_with_validation(x, y, interactions=interaction_data, train_id=train_id)

        if categorical_labels:
            y_train = to_categorical(y_train, num_classes=self.num_classes)
            y_val = to_categorical(y_val, num_classes=self.num_classes)
            y_test = to_categorical(y_test, num_classes=self.num_classes)

        if padding:
            x_train, x_val, x_test = self.create_input_tensors_pad_with_val(x_train, x_val, x_test)

        if flat:
            x_train, x_val, x_test = self.create_input_tensors_flat_with_validation(x_train, x_val, x_test)

        return [np.array(x) for x in x_train], [np.array(x) for x in x_val], [np.array(x) for x in x_test], y_train, y_val, y_test

    def split_deepface_train_val_test(self, drug_data: list[TrainingDrugDataDTO], interaction_data: list[TrainingDrugInteractionDTO], train_id: int,
                                      categorical_labels: bool = True, mean_of_text_embeddings: bool = True,
                                      pca_generating: bool = False, as_ndarray: bool = True):

        self.num_classes = len(set(item.interaction_type for item in interaction_data))

        drug_ids = [drug.drug_id for drug in drug_data]

        x = self.generate_deepface_x_values(drug_data, interaction_data, drug_ids, mean_of_text_embeddings=mean_of_text_embeddings,
                                            pca_generating=pca_generating)
        y = np.array([item.interaction_type for item in interaction_data])  # Extract labels

        x_train, x_val, x_test, y_train, y_val, y_test = self.split_on_interactions_with_validation(x, y, interactions=interaction_data, train_id=train_id)

        if categorical_labels:
            y_train = to_categorical(y_train, num_classes=self.num_classes).astype(np.int16)
            y_val = to_categorical(y_val, num_classes=self.num_classes).astype(np.int16)
            y_test = to_categorical(y_test, num_classes=self.num_classes).astype(np.int16)

        if as_ndarray:
            return [np.array(x) for x in x_train], [np.array(x) for x in x_val], [np.array(x) for x in x_test], y_train, y_val, y_test
            # return [np.array(x, dtype=np.float16) for x in x_train], [np.array(x, dtype=np.float16) for x in x_test], y_train, y_test
        else:
            return x_train, x_val, x_test, y_train, y_val, y_test

    def fold_on_interaction(self, drug_data: list[TrainingDrugDataDTO], interaction_data: list[TrainingDrugInteractionDTO], train_id: int,
                            categorical_labels: bool = True, padding: bool = False, flat: bool = False, mean_of_text_embeddings: bool = True,
                            pca_generating: bool = False, pca_components: int = None):

        self.num_classes = len(set(item.interaction_type for item in interaction_data))

        drug_ids = [drug.drug_id for drug in drug_data]

        x = self.generate_x_values(drug_data, interaction_data, drug_ids, mean_of_text_embeddings=mean_of_text_embeddings, pca_generating=pca_generating,
                                   pca_components=pca_components)
        y = np.array([item.interaction_type for item in interaction_data])  # Extract labels

        for data in self.fold_on_interactions(x, y, interactions=interaction_data, train_id=train_id):

            x_train = data["x_train"]
            x_test = data["x_test"]
            y_train = data["y_train"]
            y_test = data["y_test"]

            if categorical_labels:
                y_train = to_categorical(y_train, num_classes=self.num_classes)
                y_test = to_categorical(y_test, num_classes=self.num_classes)

            if padding:
                x_train, x_test = self.create_input_tensors_pad(x_train, x_test)

            if flat:
                x_train, x_test = self.create_input_tensors_flat(x_train, x_test)

            yield x_train, x_test, y_train, y_test

    def fold_on_interaction_deepface(self, drug_data: list[TrainingDrugDataDTO], interaction_data: list[TrainingDrugInteractionDTO], train_id: int,
                                     categorical_labels: bool = True, mean_of_text_embeddings: bool = True,
                                     pca_generating: bool = False, as_ndarray: bool = True):

        self.num_classes = len(set(item.interaction_type for item in interaction_data))

        drug_ids = [drug.drug_id for drug in drug_data]

        x = self.generate_deepface_x_values(drug_data, interaction_data, drug_ids, mean_of_text_embeddings=mean_of_text_embeddings,
                                            pca_generating=pca_generating)
        y = np.array([item.interaction_type for item in interaction_data])  # Extract labels

        for data in self.fold_on_interactions(x, y, interactions=interaction_data, train_id=train_id):
            x_train = data["x_train"]
            x_test = data["x_test"]
            y_train = data["y_train"]
            y_test = data["y_test"]

            if categorical_labels:
                y_train = to_categorical(y_train, num_classes=self.num_classes).astype(np.int16)
                y_test = to_categorical(y_test, num_classes=self.num_classes).astype(np.int16)

            if as_ndarray:
                yield [np.array(x) for x in x_train], [np.array(x) for x in x_test], y_train, y_test
                # return [np.array(x, dtype=np.float16) for x in x_train], [np.array(x, dtype=np.float16) for x in x_test], y_train, y_test
            else:
                yield x_train, x_test, y_train, y_test

    def generate_x_values(self, drug_data: list[TrainingDrugDataDTO], interactions: list[TrainingDrugInteractionDTO], train_drug_ids: list[int],
                          mean_of_text_embeddings: bool = True, pca_generating: bool = False, pca_components: int = None):
        x_output = []

        for data_set_index in tqdm(range(len(drug_data[0].train_values)), "Find X values per interactions...."):

            output_interactions = []

            if isinstance(drug_data[0].train_values[data_set_index].values, dict):
                drug_dict = {drug.drug_id: [value for key, value in drug.train_values[data_set_index].values.items() if key in train_drug_ids] for drug in
                             drug_data}

                if pca_generating:
                    drug_dict = self.calculate_pca(drug_dict, pca_components)

                for interaction in interactions:
                    drug_1_values = drug_dict.get(interaction.drug_1)
                    drug_2_values = drug_dict.get(interaction.drug_2)

                    if isinstance(drug_1_values, np.ndarray):
                        output_interactions.append(np.hstack((drug_1_values, drug_2_values)))
                    else:
                        output_interactions.append(drug_1_values + drug_2_values)

            else:
                if mean_of_text_embeddings:
                    drug_dict = {drug.drug_id: np.mean(drug.train_values[data_set_index].values, axis=1) for drug in drug_data}
                else:
                    drug_dict = {drug.drug_id: drug.train_values[data_set_index].values for drug in drug_data}

                for interaction in interactions:
                    drug_1_values = drug_dict.get(interaction.drug_1)
                    drug_2_values = drug_dict.get(interaction.drug_2)

                    output_interactions.append(drug_1_values + drug_2_values)

            x_output.append(output_interactions)

        if interactions[0].interaction_description:
            for interaction in interactions:
                x_output.append(interaction.interaction_description)

        return x_output

    def split_cnn_train_test(self, drug_data: list[TrainingDrugDataDTO], interaction_data: list[TrainingDrugInteractionDTO], train_id: int,
                             categorical_labels: bool = True):

        self.num_classes = len(set(item.interaction_type for item in interaction_data))
        drug_ids = [drug.drug_id for drug in drug_data]

        x = self.generate_cnn_x_values(drug_data, interaction_data, drug_ids)
        y = np.array([item.interaction_type for item in interaction_data])  # Extract labels

        x_train, x_test, x_val, y_val, y_train, y_test = self.split_cnn_data_on_interactions(x, y, interactions=interaction_data, train_id=train_id)

        if categorical_labels:
            y_train = to_categorical(y_train, num_classes=self.num_classes).astype(np.int16)
            y_val = to_categorical(y_val, num_classes=self.num_classes).astype(np.int16)
            y_test = to_categorical(y_test, num_classes=self.num_classes).astype(np.int16)

        return x_train, x_test, x_val, y_val, y_train, y_test

    def fold_cnn_on_interaction(self, drug_data: list[TrainingDrugDataDTO], interaction_data: list[TrainingDrugInteractionDTO], train_id: int,
                                categorical_labels: bool = True):

        self.num_classes = len(set(item.interaction_type for item in interaction_data))
        drug_ids = [drug.drug_id for drug in drug_data]

        x = self.generate_cnn_x_values(drug_data, interaction_data, drug_ids)
        y = np.array([item.interaction_type for item in interaction_data])  # Extract labels

        for data in self.fold_cnn_on_interactions(x, y, interactions=interaction_data, train_id=train_id):

            x_train = data["x_train"]
            x_test = data["x_test"]
            y_train = data["y_train"]
            y_test = data["y_test"]

            if categorical_labels:
                y_train = to_categorical(y_train, num_classes=self.num_classes)
                y_test = to_categorical(y_test, num_classes=self.num_classes)

            yield np.array(x_train), np.array(x_test), y_train, y_test

    @staticmethod
    def generate_cnn_x_values(drug_data: list[TrainingDrugDataDTO], interactions: list[TrainingDrugInteractionDTO], drug_ids: list):
        x_output = []

        def process_features(drug: TrainingDrugDataDTO, feature_count: int):
            output = [[0] * feature_count for _ in range(len(drug_ids))]

            for idx_f, f in enumerate(drug.train_values):
                values = list([v for k, v in f.values.items() if k in drug_ids])
                for idx_d in range(len(drug_ids)):
                    output[idx_d][idx_f] = values[idx_d]

            return output

        num_features = len(drug_data[0].train_values)

        drug_dict = {drug.drug_id: process_features(drug, num_features) for drug in tqdm(drug_data, "Process Drug Features...")}

        for interaction in tqdm(interactions, "Find X values per interactions...."):
            drug_1 = drug_dict.get(interaction.drug_1)
            drug_2 = drug_dict.get(interaction.drug_2)

            x_output.append([drug_1, drug_2])

        return x_output

    def generate_deepface_x_values(self, drug_data: list[TrainingDrugDataDTO], interactions: list[TrainingDrugInteractionDTO], train_drug_ids: list[int],
                                   mean_of_text_embeddings: bool = True, pca_generating: bool = False):
        x_output_1 = []
        x_output_2 = []
        interaction_output = []

        for data_set_index in tqdm(range(len(drug_data[0].train_values)), "Find X values per interactions...."):

            output_interactions_1 = []
            output_interactions_2 = []

            if isinstance(drug_data[0].train_values[data_set_index].values, dict):
                drug_dict = {drug.drug_id: [value for key, value in drug.train_values[data_set_index].values.items() if key in train_drug_ids] for drug in
                             drug_data}

                if pca_generating:
                    drug_dict = self.calculate_pca(drug_dict)

                for interaction in interactions:
                    drug_1_values = drug_dict.get(interaction.drug_1)
                    drug_2_values = drug_dict.get(interaction.drug_2)

                    output_interactions_1.append(drug_1_values)
                    output_interactions_2.append(drug_2_values)

            else:
                if mean_of_text_embeddings and drug_data[0].train_values[data_set_index].category.data_type == str:
                    drug_dict = {drug.drug_id: np.mean(drug.train_values[data_set_index].values, axis=1) for drug in drug_data}
                else:
                    drug_dict = {drug.drug_id: drug.train_values[data_set_index].values for drug in drug_data}

                    if not mean_of_text_embeddings and drug_data[0].train_values[data_set_index].category.data_type == str:
                        max_n = max(value.shape[1] for value in drug_dict.values())

                        drug_dict = {drug_id: np.pad(value, ((0, 0), (0, max_n - value.shape[1])), mode='constant') for drug_id, value in drug_dict.items()}

                for interaction in interactions:
                    drug_1_values = drug_dict.get(interaction.drug_1)
                    drug_2_values = drug_dict.get(interaction.drug_2)

                    output_interactions_1.append(drug_1_values)
                    output_interactions_2.append(drug_2_values)

            x_output_1.append(output_interactions_1)
            x_output_2.append(output_interactions_2)

        if interactions[0].interaction_description:
            for interaction in interactions:
                interaction_output.append(interaction.interaction_description)

            return x_output_1 + x_output_2 + [interaction_output]

        return x_output_1 + x_output_2

    @staticmethod
    def calculate_pca(drug_dict: dict, pca_components: int = None) -> dict:

        df = pd.DataFrame.from_dict(drug_dict)

        pca_components = pca_components if pca_components is not None else 50

        pca = PCA(n_components=pca_components)

        x = df.to_numpy()
        x_transformed = pca.fit_transform(x)

        transformed_dict = {key: x_transformed[i, :] for i, key in enumerate(drug_dict.keys())}

        return transformed_dict

    def split_on_interactions(self, x, y, interactions: list[TrainingDrugInteractionDTO], train_id: int = None, base_seed_train_id: int = None):
        x_train = [[] for _ in range(len(x))]
        x_test = [[] for _ in range(len(x))]

        y_train = []
        y_test = []

        if not base_seed_train_id:

            stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            for train_index, test_index in stratified_split.split(x[0], y):
                for i in tqdm(range(len(x)), "Splitting data"):
                    x_train[i] = [x[i][idx] for idx in train_index]
                    x_test[i] = [x[i][idx] for idx in test_index]

                y_train, y_test = y[train_index], y[test_index]
                train_interaction_ids, test_interaction_ids = [interactions[i].id for i in train_index], [interactions[i].id for i in test_index]

                self.save_seed_data(train_id, train_interaction_ids, None, test_interaction_ids)

        else:
            with open(f'seeds/{base_seed_train_id}.json', 'r') as file:
                base_seed = json.load(file)

            for idx, interaction in enumerate(interactions):
                for i in tqdm(range(len(x)), "Splitting data"):
                    if interaction.id in base_seed["train_interaction_ids"]:
                        x_train[i].append(x[i][idx])
                    elif interaction.id in base_seed["test_interaction_ids"]:
                        x_test[i].append(x[i][idx])
                    else:
                        raise

            self.save_seed_data(train_id, base_seed["train_interaction_ids"], None, base_seed["test_interaction_ids"])

        return x_train, x_test, y_train, y_test

    def split_cnn_data_on_interactions(self, x, y, interactions: list[TrainingDrugInteractionDTO], train_id: int = None):
        x_train = [[] for _ in range(len(x))]
        x_temp = [[] for _ in range(len(x))]
        x_val = [[] for _ in range(len(x))]
        x_test = [[] for _ in range(len(x))]

        y_train = []
        y_temp = []
        y_val = []
        y_test = []

        train_interaction_ids = []
        temp_interaction_ids = []
        val_interaction_ids = []
        test_interaction_ids = []

        stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
        for train_index, temp_index in stratified_split.split(x, y):
            x_train = [x[idx] for idx in train_index]
            x_temp = [x[idx] for idx in temp_index]

            y_train, y_temp = y[train_index], y[temp_index]
            train_interaction_ids, temp_interaction_ids = [interactions[i].id for i in train_index], [interactions[i].id for i in temp_index]

        stratified_split_temp = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        for val_index, test_index in stratified_split_temp.split(x_temp, y_temp):
            x_val = [x_temp[idx] for idx in val_index]
            x_test = [x_temp[idx] for idx in test_index]

            y_val, y_test = y_temp[val_index], y_temp[test_index]

            val_interaction_ids, test_interaction_ids = [temp_interaction_ids[i] for i in val_index], [temp_interaction_ids[i] for i in test_index]

        self.save_seed_data(train_id, train_interaction_ids, val_interaction_ids, test_interaction_ids)

        return np.array(x_train), np.array(x_val), np.array(x_test), y_train, y_val, y_test

    def split_on_interactions_with_validation(self, x, y, interactions: list[TrainingDrugInteractionDTO], train_id: int = None, base_seed_train_id: int = None):
        x_train = [[] for _ in range(len(x))]
        x_temp = [[] for _ in range(len(x))]
        x_val = [[] for _ in range(len(x))]
        x_test = [[] for _ in range(len(x))]

        y_train = []
        y_temp = []
        y_val = []
        y_test = []

        train_interaction_ids = []
        temp_interaction_ids = []
        val_interaction_ids = []
        test_interaction_ids = []

        if not base_seed_train_id:

            stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
            for train_index, temp_index in stratified_split.split(x[0], y):
                for i in tqdm(range(len(x)), "Splitting data"):
                    x_train[i] = [x[i][idx] for idx in train_index]
                    x_temp[i] = [x[i][idx] for idx in temp_index]

                y_train, y_temp = y[train_index], y[temp_index]

                train_interaction_ids = [interactions[i].id for i in train_index]
                temp_interaction_ids = [interactions[i].id for i in temp_index]

            stratified_split_temp = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
            for val_index, test_index in stratified_split_temp.split(x_temp[0], y_temp):
                for i in tqdm(range(len(x)), "Splitting data"):
                    x_val[i] = [x_temp[i][idx] for idx in val_index]
                    x_test[i] = [x_temp[i][idx] for idx in test_index]

                y_val, y_test = y_temp[val_index], y_temp[test_index]

                val_interaction_ids = [temp_interaction_ids[i] for i in val_index]
                test_interaction_ids = [temp_interaction_ids[i] for i in test_index]

            self.save_seed_data(train_id, train_interaction_ids, val_interaction_ids, test_interaction_ids)

        else:
            with open(f'seeds/{base_seed_train_id}.json', 'r') as file:
                base_seed = json.load(file)

            for idx, interaction in enumerate(interactions):
                for i in tqdm(range(len(x)), "Splitting data"):
                    if interaction.id in base_seed["train_interaction_ids"]:
                        x_train[i].append(x[i][idx])
                    if interaction.id in base_seed["val_interaction_ids"]:
                        x_val[i].append(x[i][idx])
                    elif interaction.id in base_seed["test_interaction_ids"]:
                        x_test[i].append(x[i][idx])
                    else:
                        raise

            self.save_seed_data(train_id, base_seed["train_interaction_ids"], base_seed["val_interaction_ids"], base_seed["test_interaction_ids"])

        return x_train, x_val, x_test, y_train, y_val, y_test

    def fold_on_interactions(self, x, y, interactions: list, train_id: int = None):
        # Containers for storing folds
        fold_data = []
        interaction_ids = [interaction.id for interaction in interactions]

        # Stratified KFold setup
        skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=42)

        for fold, (train_index, test_index) in enumerate(skf.split(x[0], y)):
            # Prepare fold-specific data containers
            x_train = [[] for _ in range(len(x))]
            x_test = [[] for _ in range(len(x))]

            y_train, y_test = y[train_index], y[test_index]

            train_interaction_ids = [interaction_ids[i] for i in train_index]
            test_interaction_ids = [interaction_ids[i] for i in test_index]

            # Populate training and validation data for each feature set
            for i in range(len(x)):
                x_train[i] = [x[i][idx] for idx in train_index]
                x_test[i] = [x[i][idx] for idx in test_index]

            # Save fold data to list
            fold_data.append({
                'fold': fold + 1,
                'x_train': x_train,
                'x_test': x_test,
                'y_train': y_train,
                'y_test': y_test,
                'train_interaction_ids': train_interaction_ids,
                'test_interaction_ids': test_interaction_ids
            })

        if train_id:
            self.save_seed_fold_data(train_id, fold_data)

        return fold_data

    def fold_cnn_on_interactions(self, x, y, interactions: list, train_id: int = None):
        # Containers for storing folds
        fold_data = []
        interaction_ids = [interaction.id for interaction in interactions]

        # Stratified KFold setup
        skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=42)

        for fold, (train_index, test_index) in enumerate(skf.split(x, y)):
            # Populate training and validation data for each feature set
            x_train = [x[idx] for idx in train_index]
            x_test = [x[idx] for idx in test_index]

            y_train, y_test = y[train_index], y[test_index]

            train_interaction_ids = [interaction_ids[i] for i in train_index]
            test_interaction_ids = [interaction_ids[i] for i in test_index]

            # Save fold data to list
            fold_data.append({
                'fold': fold + 1,
                'x_train': x_train,
                'x_test': x_test,
                'y_train': y_train,
                'y_test': y_test,
                'train_interaction_ids': train_interaction_ids,
                'test_interaction_ids': test_interaction_ids
            })

        if train_id:
            self.save_seed_fold_data(train_id, fold_data)

        return fold_data

    @staticmethod
    def save_seed_data(train_id, train_interaction_ids, val_interaction_ids, test_interaction_ids):
        seed_data = {
            "train_interaction_ids": train_interaction_ids,
            "val_interaction_ids": val_interaction_ids,
            "test_interaction_ids": test_interaction_ids
        }

        file_path = f'seeds/{train_id}.json'

        with open(file_path, 'w') as json_file:
            json.dump(seed_data, json_file)

    @staticmethod
    def save_seed_fold_data(train_id, fold_data):
        seed_data = [{
            "fold": f["fold"],
            "train_interaction_ids": f["train_interaction_ids"],
            "test_interaction_ids": f["test_interaction_ids"]
        } for f in fold_data]

        file_path = f'seeds/{train_id}.json'

        with open(file_path, 'w') as json_file:
            json.dump(seed_data, json_file)

    # endregion
