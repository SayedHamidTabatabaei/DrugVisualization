from sklearn import svm

from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from common.helpers import loss_helper
from core.models.training_parameter_model import TrainingParameterModel
from core.repository_models.training_data_dto import TrainingDataDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.SVM


class SvmTrainService(TrainBaseService):

    def train(self, parameters: TrainingParameterModel, data: list[list[TrainingDataDTO]]) -> TrainingSummaryDTO:

        x_train, x_test, y_train, y_test = super().split_train_test(data, False)

        x_train_flat, x_test_flat = super().create_input_tensors_flat(x_train, x_test)

        if parameters.class_weight:
            print('Class weight!')
            class_weight = loss_helper.get_class_weights(y_train)

            clf = svm.SVC(kernel='linear', class_weight=class_weight)
        else:
            clf = svm.SVC(kernel='linear')

        print('Start Fit')
        clf.fit(x_train_flat, y_train)

        print('Start Evaluate')
        evaluations = super().calculate_evaluation_metrics(clf, x_test_flat, y_test, True)

        if data is not None:
            evaluations.data_report = self.get_data_report_split(data[0], y_train, y_test, True)

        return evaluations
