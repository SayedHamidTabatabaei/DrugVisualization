from sklearn import svm

from businesses.trains.models.train_base_model import TrainBaseModel
from common.helpers import loss_helper
from core.models.training_params import TrainingParams
from core.repository_models.training_drug_interaction_dto import TrainingDrugInteractionDTO
from core.repository_models.training_summary_dto import TrainingSummaryDTO


class SVMModel(TrainBaseModel):
    def __init__(self, train_id: int, num_classes: int, interaction_data: list[TrainingDrugInteractionDTO], training_params: TrainingParams):
        super().__init__(train_id, num_classes)
        self.interaction_data = interaction_data
        self.training_params = training_params

    def fit_model(self, x_train, y_train, x_val, y_val, x_test, y_test) -> TrainingSummaryDTO:

        if x_val or y_val:
            print("In this algorithm, it doesn't use x_val and y_val!")

        if self.training_params.class_weight:
            print('Class weight!')
            class_weight = loss_helper.get_class_weights(y_train)

            model = svm.SVC(kernel='linear', class_weight=class_weight)
        else:
            model = svm.SVC(kernel='linear')

        model.fit(x_train, y_train)

        result = self.calculate_evaluation_metrics(model, x_test, y_test)

        result.model_info = self.get_model_info(model)

        result.data_report = self.get_data_report_split(self.interaction_data, y_train, y_test)

        return result
