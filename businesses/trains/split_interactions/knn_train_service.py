from sklearn.neighbors import KNeighborsClassifier

from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from core.models.training_parameter_models.split_interaction_similarities_training_parameter_model import SplitInteractionSimilaritiesTrainingParameterModel
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.KNN


class KnnTrainService(TrainBaseService):

    def train(self, parameters: SplitInteractionSimilaritiesTrainingParameterModel) -> TrainingSummaryDTO:

        x_train, x_test, y_train, y_test = super().split_train_test(parameters.drug_data, parameters.interaction_data,
                                                                    train_id=parameters.train_id, padding=True, flat=True)

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(x_train, y_train)

        evaluations = self.calculate_evaluation_metrics(knn, x_test, y_test)

        return evaluations