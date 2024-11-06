from sklearn.linear_model import LogisticRegression

from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from common.helpers import loss_helper
from core.models.training_parameter_models.split_interaction_similarities_training_parameter_model import SplitInteractionSimilaritiesTrainingParameterModel
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.Drug_LR or TrainModel.Drug_LR_Test


class DrugLrTrainService(TrainBaseService):

    def __init__(self, category, compare_train_test: bool = True):
        super().__init__(category)
        self.compare_train_test = compare_train_test

    def train(self, parameters: SplitInteractionSimilaritiesTrainingParameterModel) -> TrainingSummaryDTO:
        results = []

        for x_train, x_test, y_train, y_test in super().manual_k_fold_train_test_data(parameters.drug_data, parameters.interaction_data,
                                                                                      padding=True, flat=True, compare_train_test=self.compare_train_test):

            if parameters.class_weight:
                print('Class weight!')
                class_weight = loss_helper.get_class_weights(y_train)

                model = LogisticRegression(max_iter=10000, class_weight=class_weight)
            else:
                model = LogisticRegression(max_iter=10000)

            model.fit(x_train, y_train)

            result = self.calculate_evaluation_metrics(model, x_test, y_test, True)

            result.model_info = self.get_model_info(model)

            result.data_report = self.get_data_report_split(parameters.interaction_data, y_train, y_test, True)

            results.append(result)

        return super().calculate_fold_results(results)
