from sklearn.neighbors import KNeighborsClassifier

from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from core.models.training_parameter_models.split_interaction_similarities_training_parameter_model import SplitInteractionSimilaritiesTrainingParameterModel
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.Fold_KNN


class FoldKnnTrainService(TrainBaseService):

    def train(self, parameters: SplitInteractionSimilaritiesTrainingParameterModel) -> TrainingSummaryDTO:
        results = []

        for x_train, x_test, y_train, y_test in super().fold_on_interaction(parameters.drug_data, parameters.interaction_data,
                                                                            train_id=parameters.train_id, padding=True,
                                                                            flat=True):

            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(x_train, y_train)

            result = self.calculate_evaluation_metrics(knn, x_test, y_test)

            result.model_info = self.get_model_info(knn)

            result.data_report = self.get_data_report_split(parameters.interaction_data, y_train, y_test)

            results.append(result)

        return super().calculate_fold_results(results)
