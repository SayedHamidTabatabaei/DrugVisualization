import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.preprocessing import label_binarize

from businesses.trains.train_plan_base import TrainPlanBase
from common.enums.train_models import TrainModel
from core.repository_models.training_data_dto import TrainingDataDTO

train_model = TrainModel.SimpleOneInput


class TrainPlan2(TrainPlanBase):

    def train(self, data: list[list[TrainingDataDTO]], train_id):

        data = data[0]

        # Prepare input for the model
        x_pairs = np.array([np.concatenate((item.reduction_values_1, item.reduction_values_2)) for item in data])

        # Example labels (replace this with your actual interaction data)
        y = np.array([item.interaction_type for item in data])

        x_train = []
        x_test = []
        y_train = []
        y_test = []

        stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in stratified_split.split(x_pairs, y):
            x_train, x_test = x_pairs[train_index], x_pairs[test_index]
            y_train, y_test = y[train_index], y[test_index]

        # Define the model
        model = Sequential()

        # Input layer
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))

        # Hidden layers
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(32, activation='relu'))

        # Output layer with softmax
        model.add(Dense(65, activation='softmax'))

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

        loss, accuracy = model.evaluate(x_test, y_test)
        print(f'Old Accuracy: {accuracy * 100:.2f}%')

        # Predictions
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred_classes)
        f1 = f1_score(y_test, y_pred_classes, average='weighted')

        # For multi-class AUC and AUPR, you need to binarize the labels
        y_test_bin = label_binarize(y_test, classes=range(65))
        auc = roc_auc_score(y_test_bin, y_pred, average='weighted', multi_class='ovr')
        aupr = average_precision_score(y_test_bin, y_pred, average='weighted')

        # Binarize the labels for AUC and AUPR calculation
        y_test_bin = label_binarize(y_test, classes=range(65))
        # results_per_labels: list[TrainingResultDetailDTO] = []
        #
        # for i in range(65):
        #     class_accuracy = accuracy_score(y_test == i, y_pred_classes == i)
        #
        #     class_f1 = f1_score(y_test == i, y_pred_classes == i, average='binary')
        #
        #     class_auc = roc_auc_score(y_test_bin[:, i], y_pred[:, i])
        #
        #     precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred[:, i])
        #     class_aupr = average_precision_score(y_test_bin[:, i], y_pred[:, i])
        #
        #     results_per_labels.append(TrainingResultDetailDTO(training_label=i,
        #                                                       f1_score=class_f1,
        #                                                       accuracy=class_accuracy,
        #                                                       auc=class_auc,
        #                                                       aupr=class_aupr))
        #
        # return f1, accuracy, auc, aupr, results_per_labels
