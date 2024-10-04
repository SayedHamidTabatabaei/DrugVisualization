import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from common.helpers import loss_helper
from core.models.training_parameter_model import TrainingParameterModel
from core.repository_models.training_data_dto import TrainingDataDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.RF


class RfTrainService(TrainBaseService):

    def train(self, parameters: TrainingParameterModel, data: list[list[TrainingDataDTO]]) -> TrainingSummaryDTO:

        x_train, x_test, y_train, y_test = super().split_train_test(data, False)

        x_train = np.array([np.concatenate([x_train[j][i] for j in range(len(x_train))]) for i in tqdm(range(len(x_train[0])), "Flat train data!")])
        x_test = np.array([np.concatenate([x_test[j][i] for j in range(len(x_test))]) for i in tqdm(range(len(x_test[0])), "Flat test data!")])

        if parameters.class_weight:
            print('Class weight!')
            class_weight = loss_helper.get_class_weights(y_train)

            rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight)
        else:
            rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

        print('Fit!')
        rf_clf.fit(x_train, y_train)

        print('Evaluate!')
        result = self.calculate_evaluation_metrics(rf_clf, x_test, y_test, True)

        if data is not None:
            result.data_report = self.get_data_report_split(data[0], y_train, y_test, True)

        return result
