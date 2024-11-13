from sklearn.neighbors import KNeighborsClassifier

from businesses.trains.models.train_base_model import TrainBaseModel
from core.repository_models.training_drug_interaction_dto import TrainingDrugInteractionDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO


class KNNModel(TrainBaseModel):
    def __init__(self, train_id: int, num_classes: int, interaction_data: list[TrainingDrugInteractionDTO]):
        super().__init__(train_id, num_classes)
        self.interaction_data = interaction_data

    def fit_model(self, x_train, y_train, x_val, y_val, x_test, y_test) -> TrainingSummaryDTO:

        if x_val or y_val:
            print("In this algorithm, it doesn't use x_val and y_val!")

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(x_train, y_train)

        result = self.calculate_evaluation_metrics(knn, x_test, y_test)

        result.model_info = self.get_model_info(knn)

        result.data_report = self.get_data_report_split(self.interaction_data, y_train, y_test)

        return result
