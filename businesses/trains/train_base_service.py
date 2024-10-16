import contextlib
import io
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_score, \
    recall_score, log_loss
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from sklearn.preprocessing import label_binarize
# noinspection PyUnresolvedReferences
from tensorflow.keras.preprocessing.sequence import pad_sequences
# noinspection PyUnresolvedReferences
from tensorflow.keras.preprocessing.text import Tokenizer
# noinspection PyUnresolvedReferences
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from common.enums.train_models import TrainModel
from common.enums.training_result_type import TrainingResultType
from common.helpers import loss_helper
from core.models.data_params import DataParams
from core.models.training_parameter_base_model import TrainingParameterBaseModel
from core.models.training_params import TrainingParams
from core.repository_models.training_drug_data_dto import TrainingDrugDataDTO
from core.repository_models.training_drug_interaction_dto import TrainingDrugInteractionDTO
from core.repository_models.training_result_detail_summary_dto import TrainingResultDetailSummaryDTO
from core.repository_models.training_result_summary_dto import TrainingResultSummaryDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO


class TrainBaseService:
    category: TrainModel

    def __init__(self, category: TrainModel):
        self.category = category
        self.num_classes = 65
        self.num_folds = 5

    def save_image(self, train_id):
        pass

    def train(self, parameters: TrainingParameterBaseModel) -> TrainingSummaryDTO:
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
                                  data_report=None,
                                  model_info=None)

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
    def pad_sequences(data, maxlen=None, padding_value='0'):
        return tf.keras.preprocessing.sequence.pad_sequences(data, maxlen=maxlen, padding='post', value=padding_value)

    @staticmethod
    def create_tf_dataset(x_train, y_train, batch_size=256):
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def fit_dnn_model(self, data_params: DataParams, training_params: TrainingParams, model, interactions: (list[TrainingDrugInteractionDTO] | None) = None) \
            -> TrainingSummaryDTO:
        if training_params.class_weight:
            class_weights = loss_helper.get_class_weights(data_params.y_train)
        else:
            class_weights = None

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

        result.model_info = self.get_model_info(model)

        if interactions is not None:
            result.data_report = self.get_data_report_split(interactions, data_params.y_train, data_params.y_test)

        return result

    @staticmethod
    def get_model_info(model):

        summary_str = io.StringIO()
        with contextlib.redirect_stdout(summary_str):
            model.summary()
        model_summary = summary_str.getvalue()

        # Calculate parameters
        total_trainable_params = int(np.sum([np.prod(v.shape.as_list()) for v in model.trainable_weights]))
        total_params = model.count_params()

        # Create dictionary to save as JSON
        model_info = {
            "model_summary": model_summary,
            "total_trainable_params": total_trainable_params,
            "total_params": total_params
        }

        return model_info

    @staticmethod
    def get_data_report(interactions: list[TrainingDrugInteractionDTO]):
        df = pd.DataFrame([vars(interaction) for interaction in interactions])

        grouped_counts = df.groupby('interaction_type').size()

        grouped_counts_dict = grouped_counts.reset_index().to_dict(orient='records')

        return [{g['interaction_type']: g[0]} for g in grouped_counts_dict]

    @staticmethod
    def get_data_report_by_labels(labels: list[int], is_labels_categorical: bool = False):
        indices = np.argmax(labels, axis=1) if not is_labels_categorical else labels

        counts = np.bincount(indices, minlength=65)

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

    # region split data
    def k_fold_train_test_data(self, drug_data: list[TrainingDrugDataDTO], interaction_data: list[TrainingDrugInteractionDTO], padding: bool = False,
                               flat: bool = False):

        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)

        drug_ids = [drug.drug_id for drug in drug_data]

        for train_drug_indices, test_drug_indices in kf.split(drug_ids):
            # Create sets of drug IDs for training and testing sets
            train_drug_ids = {drug_ids[i] for i in train_drug_indices}
            test_drug_ids = {drug_ids[i] for i in test_drug_indices}

            # Filter interaction_data based on the defined conditions
            train_interactions = [
                interaction for interaction in interaction_data
                if interaction.drug_1 in train_drug_ids and interaction.drug_2 in train_drug_ids
            ]

            test_interactions = [
                interaction for interaction in interaction_data
                if interaction.drug_1 in test_drug_ids or interaction.drug_2 in test_drug_ids
            ]

            y_train = [i.interaction_type for i in train_interactions]
            y_test = [i.interaction_type for i in test_interactions]

            x_train = self.generate_x_values(drug_data, train_interactions, train_drug_ids)
            x_test = self.generate_x_values(drug_data, test_interactions, train_drug_ids)

            if padding:
                x_train, x_test = self.create_input_tensors_pad(x_train, x_test)

            if flat:
                x_train, x_test = self.create_input_tensors_flat(x_train, x_test)

            yield x_train, x_test, y_train, y_test

    def split_train_test(self, drug_data: list[TrainingDrugDataDTO], interaction_data: list[TrainingDrugInteractionDTO], categorical_labels: bool = True,
                         padding: bool = False, flat: bool = False):

        drug_ids = [drug.drug_id for drug in drug_data]
        x = self.generate_x_values(drug_data, interaction_data, drug_ids)
        y = np.array([item.interaction_type for item in interaction_data])  # Extract labels

        x_train = [[] for _ in range(len(x))]
        x_test = [[] for _ in range(len(x))]

        y_train = []
        y_test = []

        stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in stratified_split.split(x[0], y):
            for i in tqdm(range(len(x)), "Splitting data"):
                x_train[i] = [x[i][idx] for idx in train_index]
                x_test[i] = [x[i][idx] for idx in test_index]

            y_train, y_test = y[train_index], y[test_index]

        if categorical_labels:
            y_train = to_categorical(y_train, num_classes=self.num_classes)
            y_test = to_categorical(y_test, num_classes=self.num_classes)

            if padding:
                x_train, x_test = self.create_input_tensors_pad(x_train, x_test)

            if flat:
                x_train, x_test = self.create_input_tensors_flat(x_train, x_test)

        return x_train, x_test, y_train, y_test

    @staticmethod
    def generate_x_values(drug_data: list[TrainingDrugDataDTO], interactions: list[TrainingDrugInteractionDTO], train_drug_ids):
        x_output = []

        for data_set_index in tqdm(range(len(drug_data[0].train_values)), "Find X values per interactions...."):

            output_interactions = []

            if isinstance(drug_data[0].train_values[data_set_index].values, dict):
                drug_dict = {drug.drug_id: [value for key, value in drug.train_values[data_set_index].values.items() if key in train_drug_ids] for drug in
                             drug_data}

                for interaction in interactions:
                    drug_1_values = drug_dict.get(interaction.drug_1)
                    drug_2_values = drug_dict.get(interaction.drug_2)

                    output_interactions.append(drug_1_values + drug_2_values)

            else:
                drug_dict = {drug.drug_id: drug.train_values[data_set_index].values for drug in drug_data}

                for interaction in interactions:
                    drug_1_values = drug_dict.get(interaction.drug_1)
                    drug_2_values = drug_dict.get(interaction.drug_2)

                    output_interactions.append(drug_1_values + drug_2_values)

            x_output.append(output_interactions)

        return x_output

    # endregion
