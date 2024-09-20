from sklearn.neighbors import KNeighborsClassifier

from businesses.trains.train_plan_base import TrainPlanBase
from common.enums.train_models import TrainModel
from core.repository_models.training_data_dto import TrainingDataDTO
from core.repository_models.training_result_summary_dto import TrainingResultSummaryDTO

train_model = TrainModel.KNN


class TrainPlan6(TrainPlanBase):

    def train(self, data: list[list[TrainingDataDTO]], train_id: int) -> TrainingResultSummaryDTO:

        x_train, x_test, y_train, y_test = super().split_train_test(data)

        x_train_ragged, x_test_ragged = super().create_input_tensors(x_train, x_test)

        knn = KNeighborsClassifier(n_neighbors=5)
        history = knn.fit(x_train_ragged, y_train)

        evaluations = super().calculate_evaluation_metrics(x_train_ragged, x_test_ragged, y_test)

        super().plot_accuracy(history, train_id)
        super().plot_loss(history, train_id)

        # super().plot_accuracy_radial([item.accuracy for item in evaluations.training_result_details], train_id)
        # super().plot_f1_score_radial([item.f1_score for item in evaluations.training_result_details], train_id)
        # super().plot_auc_radial([item.auc for item in evaluations.training_result_details], train_id)

        return evaluations
