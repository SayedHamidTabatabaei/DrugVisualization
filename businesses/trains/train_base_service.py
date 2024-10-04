import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_score, \
    recall_score, log_loss
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from common.enums.train_models import TrainModel
from common.enums.training_result_type import TrainingResultType
from common.helpers import loss_helper
from core.models.data_params import DataParams
from core.models.training_parameter_model import TrainingParameterModel
from core.models.training_params import TrainingParams
from core.repository_models.training_data_dto import TrainingDataDTO
from core.repository_models.training_result_detail_summary_dto import TrainingResultDetailSummaryDTO
from core.repository_models.training_result_summary_dto import TrainingResultSummaryDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO


class TrainBaseService:
    category: TrainModel

    def __init__(self, category: TrainModel):
        self.category = category
        self.num_classes = 65

    def split_train_test(self, data: list[list[TrainingDataDTO]], categorical_labels: bool = True):

        # Prepare input features
        x_pairs = [[item.concat_values for item in d] for d in data]  # Extract features
        y = np.array([item.interaction_type for item in data[0]])  # Extract labels

        # Initialize empty lists
        x_train = [[] for _ in range(len(x_pairs))]
        x_test = [[] for _ in range(len(x_pairs))]

        y_train = []
        y_test = []

        stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in stratified_split.split(x_pairs[0], y):
            for i in tqdm(range(len(x_pairs)), "Splitting data"):
                x_train[i] = [x_pairs[i][idx] for idx in train_index]
                x_test[i] = [x_pairs[i][idx] for idx in test_index]

            y_train, y_test = y[train_index], y[test_index]

        if categorical_labels:
            y_train = to_categorical(y_train, num_classes=self.num_classes)
            y_test = to_categorical(y_test, num_classes=self.num_classes)

        return x_train, x_test, y_train, y_test

    #
    # def split_train_test_flatten(self, data: list[list[TrainingDataDTO]]):
    #
    #     x_pairs = [[item.concat_values for item in d] for d in data]
    #     x_flat = TrainBaseService.flatten_x_train(data)
    #
    #     y = np.array([item.interaction_type for item in data[0]])
    #
    #     x_train = [[] for _ in range(len(x_flat))]
    #     x_test = [[] for _ in range(len(x_flat))]
    #
    #     y_train = []
    #     y_test = []
    #
    #     stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    #     for train_index, test_index in stratified_split.split(x_flat[0], y):
    #         for i in tqdm(range(len(x_flat)), "Splitting data"):
    #             x_train[i] = [x_flat[i][idx] for idx in train_index]
    #             x_test[i] = [x_flat[i][idx] for idx in test_index]
    #
    #         y_train, y_test = y[train_index], y[test_index]
    #
    #     y_train = to_categorical(y_train, num_classes=self.num_classes)
    #     y_test = to_categorical(y_test, num_classes=self.num_classes)
    #
    #     return x_train, x_test, y_train, y_test

    def save_image(self, train_id):
        pass

    def train(self, parameters: TrainingParameterModel, data: list[list[TrainingDataDTO]]) -> TrainingSummaryDTO:
        pass

    def calculate_evaluation_metrics(self, model, x_test, y_test, is_labels_categorical: bool = False) -> TrainingSummaryDTO:

        training_results: list[TrainingResultSummaryDTO] = []

        # Predictions
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

        # print(f'Total Accuracy: {total_accuracy * 100:.2f}% and Accuracy on Pred: {accuracy * 100:.2f}')

        f1_score_weighted = f1_score(y_test_classes, y_pred_classes, average='weighted')
        training_results.append(TrainingResultSummaryDTO(TrainingResultType.f1_score_weighted, f1_score_weighted))

        f1_score_micor = f1_score(y_test_classes, y_pred_classes, average='micro')
        training_results.append(TrainingResultSummaryDTO(TrainingResultType.f1_score_micro, f1_score_micor))

        f1_score_macro = f1_score(y_test_classes, y_pred_classes, average='macro')
        training_results.append(TrainingResultSummaryDTO(TrainingResultType.f1_score_macro, f1_score_macro))

        # For multi-class AUC and AUPR, you need to binarize the labels
        y_test_bin = label_binarize(y_test_classes, classes=range(self.num_classes))
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
                                  data_report=None)

    @staticmethod
    def plot_accuracy(history, train_id: int):

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
    def plot_loss(history, train_id: int):

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

    # @staticmethod
    # def flatten_x_train(x_train):
    #     flattened_samples = []
    #
    #     for sample_group in x_train:
    #         for feature_vector in sample_group:
    #             flattened_samples.append(np.array(feature_vector))
    #
    #     return np.array(flattened_samples)

    @staticmethod
    def create_input_tensors_pad(x_train, x_test):

        max_len_train = max(max(len(seq) for seq in x_train[i]) for i in range(len(x_train)))
        max_len_test = max(max(len(seq) for seq in x_test[i]) for i in range(len(x_test)))
        max_len = max(max_len_train, max_len_test)

        x_train_padded = [
            [np.pad(seq, (0, max_len - len(seq)), 'constant', constant_values=0) for seq in x_train[i]]
            for i in tqdm(range(len(x_train)), "Padding train data ...")
        ]

        x_test_padded = [
            [np.pad(seq, (0, max_len - len(seq)), 'constant', constant_values=0) for seq in x_test[i]]
            for i in tqdm(range(len(x_test)), "Padding test data ...")
        ]

        x_train_padded = np.array(x_train_padded[0])
        x_test_padded = np.array(x_test_padded[0])

        return x_train_padded, x_test_padded

    @staticmethod
    def create_input_tensors_flat(x_train, x_test):
        x_train_padded, x_test_padded = TrainBaseService.create_input_tensors_pad(x_train, x_test)

        x_train_flat = x_train_padded.reshape(x_train_padded.shape[0], -1)
        x_test_flat = x_test_padded.reshape(x_test_padded.shape[0], -1)

        return x_train_flat, x_test_flat

    @staticmethod
    def pad_sequences(data, maxlen=None, padding_value='0'):
        return tf.keras.preprocessing.sequence.pad_sequences(data, maxlen=maxlen, padding='post', value=padding_value)

    @staticmethod
    def create_tf_dataset(x_train, y_train, batch_size=256):
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def fit_dnn_model(self, data_params: DataParams, training_params: TrainingParams, model, data=None) -> TrainingSummaryDTO:
        if training_params.class_weight:
            class_weights = loss_helper.get_class_weights(data_params.y_train)
        else:
            class_weights = None

        # Create the model
        model.compile(optimizer=training_params.optimizer,
                      loss=loss_helper.get_loss_function(training_params.loss, class_weights),
                      metrics=training_params.metrics)

        print('Fit data!')
        history = model.fit(data_params.x_train, data_params.y_train,
                            epochs=50, batch_size=256,
                            validation_data=(data_params.x_test, data_params.y_test),
                            class_weight=class_weights)

        result = self.calculate_evaluation_metrics(model, data_params.x_test, data_params.y_test)

        self.plot_accuracy(history, training_params.train_id)
        self.plot_loss(history, training_params.train_id)

        if data is not None:
            result.data_report = self.get_data_report_split(data[0], data_params.y_train, data_params.y_test)

        return result

    @staticmethod
    def get_data_report(data: list[TrainingDataDTO]):
        df = pd.DataFrame([vars(d) for d in data])

        grouped_counts = df.groupby('interaction_type').size()

        grouped_counts_dict = grouped_counts.reset_index().to_dict(orient='records')

        return [{g['interaction_type']: g[0]} for g in grouped_counts_dict]

    @staticmethod
    def get_data_report_by_labels(labels: list[int], is_labels_categorical: bool = False):
        indices = np.argmax(labels, axis=1) if not is_labels_categorical else labels

        counts = np.bincount(indices, minlength=65)

        return [{i: c} for i, c in enumerate(counts)]

    def get_data_report_split(self, data: list[TrainingDataDTO], y_train, y_test, is_labels_categorical: bool = False):
        return {
            "total_count": len(data),
            "train_count": len(y_train),
            "test_count": len(y_test),
            "total_report": self.get_data_report(data),
            "train_report": self.get_data_report_by_labels(y_train, is_labels_categorical),
            "test_report": self.get_data_report_by_labels(y_test, is_labels_categorical)
        }
