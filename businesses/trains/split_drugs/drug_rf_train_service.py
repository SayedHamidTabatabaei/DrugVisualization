from businesses.trains.models.rf_model import RFModel
from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from core.models.training_parameter_models.split_interaction_similarities_training_parameter_model import \
    SplitInteractionSimilaritiesTrainingParameterModel
from core.models.training_params import TrainingParams
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.Drug_RF or TrainModel.Drug_RF_Test


class DrugRfTrainService(TrainBaseService):

    def __init__(self, category, compare_train_test: bool = True, file_train_id: int = None):
        super().__init__(category, file_train_id=file_train_id)
        self.compare_train_test = compare_train_test

    def train(self, parameters: SplitInteractionSimilaritiesTrainingParameterModel) -> TrainingSummaryDTO:
        training_params = TrainingParams(train_id=parameters.train_id, optimizer='adam', loss=parameters.loss_function,
                                         class_weight=parameters.class_weight)

        results = []

        for x_train, x_test, y_train, y_test in super().manual_k_fold_train_test_data(parameters.drug_data,
                                                                                      parameters.interaction_data,
                                                                                      padding=True, flat=True,
                                                                                      compare_train_test=self.compare_train_test,
                                                                                      categorical_labels=False):
            model = RFModel(parameters.train_id, self.num_classes, parameters.interaction_data,
                            training_params=training_params)
            result = model.fit_model(x_train, y_train, x_test, y_test, x_test, y_test)

            results.append(result)

        return super().calculate_fold_results(results)
