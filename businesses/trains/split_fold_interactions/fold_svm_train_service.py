from sklearn import svm

from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from common.helpers import loss_helper
from core.models.training_parameter_models.split_interaction_similarities_training_parameter_model import SplitInteractionSimilaritiesTrainingParameterModel
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.Fold_SVM


class FoldSvmTrainService(TrainBaseService):

    def train(self, parameters: SplitInteractionSimilaritiesTrainingParameterModel) -> TrainingSummaryDTO:
        results = []

        for x_train, x_test, y_train, y_test in super().fold_on_interaction(parameters.drug_data, parameters.interaction_data,
                                                                            train_id=parameters.train_id, categorical_labels=False, padding=True,
                                                                            flat=True):

            if parameters.class_weight:
                print('Class weight!')
                class_weight = loss_helper.get_class_weights(y_train)

                svm_model = svm.SVC(kernel='linear', class_weight=class_weight)
            else:
                svm_model = svm.SVC(kernel='linear')

            print('Start Fit')
            svm_model.fit(x_train, y_train)

            print('Start Evaluate')
            result = super().calculate_evaluation_metrics(svm_model, x_test, y_test, True)

            result.model_info = self.get_model_info(svm_model)

            result.data_report = self.get_data_report_split(parameters.interaction_data, y_train, y_test, True)

            results.append(result)

        return super().calculate_fold_results(results)
