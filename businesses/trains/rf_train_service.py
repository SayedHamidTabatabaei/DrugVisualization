from sklearn.ensemble import RandomForestClassifier

from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from common.helpers import loss_helper
from core.models.training_parameter_models.split_interaction_similarities_training_parameter_model import SplitInteractionSimilaritiesTrainingParameterModel
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.RF


class RfTrainService(TrainBaseService):

    def train(self, parameters: SplitInteractionSimilaritiesTrainingParameterModel) -> TrainingSummaryDTO:

        x_train, x_test, y_train, y_test = super().split_train_test(parameters.drug_data, parameters.interaction_data, categorical_labels=False, padding=True,
                                                                    flat=True)

        if parameters.class_weight:
            print('Class weight!')
            class_weight = loss_helper.get_class_weights(y_train)

            rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight)
        else:
            rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

        print('Fit!')
        rf_clf.fit(x_train, y_train)

        print('Evaluate!')
        result = self.calculate_evaluation_metrics(rf_clf, x_test, y_test, True)

        result.data_report = self.get_data_report_split(parameters.interaction_data, y_train, y_test, True)

        return result
