from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import compute_sample_weight

from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from core.models.training_parameter_model import TrainingParameterModel
from core.repository_models.training_data_dto import TrainingDataDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.KNN


class KnnTrainService(TrainBaseService):

    def train(self, parameters: TrainingParameterModel, data: list[list[TrainingDataDTO]]) -> TrainingSummaryDTO:

        x_train, x_test, y_train, y_test = super().split_train_test(data)

        x_train_flat, x_test_flat = super().create_input_tensors_flat(x_train, x_test)

        if parameters.class_weight:
            print('Class weight!')
            sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
        else:
            sample_weight = None

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(x_train_flat, y_train, sample_weight=sample_weight)

        evaluations = super().calculate_evaluation_metrics(knn, x_test_flat, y_test)

        return evaluations
