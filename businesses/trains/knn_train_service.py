from sklearn.neighbors import KNeighborsClassifier

from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from core.models.training_parameter_models.split_interaction_similarities_training_parameter_model import SplitInteractionSimilaritiesTrainingParameterModel
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.KNN


class KnnTrainService(TrainBaseService):

    def train(self, parameters: SplitInteractionSimilaritiesTrainingParameterModel) -> TrainingSummaryDTO:

        x_train, x_test, y_train, y_test = super().split_train_test(parameters.data)

        x_train, x_test = super().create_input_tensors_flat(x_train, x_test)

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(x_train, y_train)

        evaluations = super().calculate_evaluation_metrics(knn, x_test, y_test)

        return evaluations