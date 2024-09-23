import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_score, \
    recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from common.enums.train_models import TrainModel
from core.repository_models.training_data_dto import TrainingDataDTO
from core.repository_models.training_result_detail_summary_dto import TrainingResultDetailSummaryDTO
from core.repository_models.training_result_summary_dto import TrainingResultSummaryDTO


class TrainPlanBase:
    category: TrainModel

    def __init__(self, category: TrainModel):
        self.category = category
        self.num_classes = 65

    def split_train_test(self, data: list[list[TrainingDataDTO]]):
        # Prepare input for the model
        x_pairs = [[item.concat_values for item in d] for d in data]

        # Example labels (replace this with your actual interaction data)
        y = np.array([item.interaction_type for item in data[0]])

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

        y_train = to_categorical(y_train, num_classes=self.num_classes)
        y_test = to_categorical(y_test, num_classes=self.num_classes)

        return x_train, x_test, y_train, y_test

    def save_image(self, train_id):
        pass

    def train(self, data: list[list[TrainingDataDTO]], train_id: int) -> TrainingResultSummaryDTO:
        pass

    def calculate_evaluation_metrics(self, model, x_test, y_test) -> TrainingResultSummaryDTO:

        print('Calculate loss!')
        loss, total_accuracy = model.evaluate(x_test, y_test)

        # Predictions
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        y_test_classes = np.argmax(y_test, axis=1)

        accuracy = accuracy_score(y_test_classes, y_pred_classes)
        print(f'Total Accuracy: {total_accuracy * 100:.2f}% and Accuracy on Pred: {accuracy * 100:.2f}')

        f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')

        # For multi-class AUC and AUPR, you need to binarize the labels
        y_test_bin = label_binarize(y_test_classes, classes=range(self.num_classes))
        auc = roc_auc_score(y_test_bin, y_pred, average='weighted', multi_class='ovr')
        aupr = average_precision_score(y_test_bin, y_pred, average='weighted')
        precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
        recall = recall_score(y_test_classes, y_pred_classes, average='weighted')

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

        return TrainingResultSummaryDTO(f1_score=f1,
                                        accuracy=accuracy,
                                        loss=loss,
                                        auc=auc,
                                        aupr=aupr,
                                        recall=recall,
                                        precision=precision,
                                        model=model,
                                        training_result_details=results_per_labels)

    @staticmethod
    def plot_accuracy(history, train_id: int):

        folder_name = TrainPlanBase.get_image_folder_name(train_id)

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

        folder_name = TrainPlanBase.get_image_folder_name(train_id)

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

        folder_name = TrainPlanBase.get_image_folder_name(train_id)

        plt = TrainPlanBase.plot_radial(values)

        plt.title('Accuracy')

        plot_path = os.path.join(folder_name, 'accuracies.png')
        plt.savefig(plot_path)

        plt.close()

    @staticmethod
    def plot_f1_score_radial(values, train_id: int):

        folder_name = TrainPlanBase.get_image_folder_name(train_id)

        plt = TrainPlanBase.plot_radial(values)

        plt.title('F1 Score')

        plot_path = os.path.join(folder_name, 'f1_score.png')
        plt.savefig(plot_path)

        plt.close()

    @staticmethod
    def plot_auc_radial(values, train_id: int):

        folder_name = TrainPlanBase.get_image_folder_name(train_id)

        plt = TrainPlanBase.plot_radial(values)

        plt.title('AUC')

        plot_path = os.path.join(folder_name, 'auc.png')
        plt.savefig(plot_path)

        plt.close()

    # @tf.function
    @staticmethod
    def create_input_tensors(x_train, x_test):
        start_time = time.time()
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

        # x_train_processed = []
        # x_test_processed = []
        #
        # # Preprocess SMILES data (assumed to be in the first set)
        # smiles_train = [d[0] for d in x_train[0]]
        # smiles_test = [d[0] for d in x_test[0]]
        #
        # # Convert SMILES to tensors (use batching instead of element-wise conversion)
        # smiles_train_tensor = tf.constant(smiles_train, dtype=tf.string)
        # smiles_test_tensor = tf.constant(smiles_test, dtype=tf.string)
        #
        # # Append to the processed list
        # x_train_processed.append(smiles_train_tensor)
        # x_test_processed.append(smiles_test_tensor)
        #
        # # Handle the float data (other sets)
        # for i in range(1, len(x_train)):  # Assuming the other sets are from 1 to n
        #     float_train = [d for d in x_train[i]]
        #     float_test = [d for d in x_test[i]]
        #
        #     # If the shapes are irregular, use ragged tensors
        #     if any(len(f) != len(float_train[0]) for f in float_train):
        #         x_train_processed.append(tf.ragged.constant(float_train))
        #         x_test_processed.append(tf.ragged.constant(float_test))
        #     else:
        #         x_train_processed.append(tf.constant(float_train, dtype=tf.float32))
        #         x_test_processed.append(tf.constant(float_test, dtype=tf.float32))

        x_train_processed = TrainPlanBase.pad_sequences(x_train)
        x_test_processed = TrainPlanBase.pad_sequences(x_test)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Finished time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Execution time: {execution_time} seconds")

        return x_train_processed, x_test_processed

    @staticmethod
    def pad_sequences(data, maxlen=None, padding_value='0'):
        return tf.keras.preprocessing.sequence.pad_sequences(data, maxlen=maxlen, padding='post', value=padding_value)

    @staticmethod
    def create_tf_dataset(x_train, y_train, batch_size=256):
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

