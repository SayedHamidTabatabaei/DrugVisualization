import contextlib
import io
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_score, \
    recall_score, log_loss

from common.enums.training_result_type import TrainingResultType
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

    @staticmethod
    def get_image_folder_name(train_id: int) -> str:
        folder_name = f"training_plots/{train_id}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        return folder_name

    def plot_accuracy(self, history, train_id):

        folder_name = self.get_image_folder_name(train_id)

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

    def plot_loss(self, history, train_id):

        folder_name = self.get_image_folder_name(train_id)

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
    def calculate_shapes(x_train):
        x_train_shapes = [x.shape for x in x_train]

        return x_train_shapes
