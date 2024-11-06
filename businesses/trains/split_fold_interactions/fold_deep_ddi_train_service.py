from businesses.trains.models.deep_ddi_model import DeepDDITrainModel
from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from core.models.training_parameter_models.split_drugs_test_with_train_training_parameter_model import SplitDrugsTestWithTrainTrainingParameterModel
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.Fold_Deep_DDI


class FoldDeepDDITrainService(TrainBaseService):

    def __init__(self, category, compare_train_test: bool = True):
        super().__init__(category)
        self.drug_channels: int = 256
        self.hidden_channels: int = 2048
        self.hidden_layers_num: int = 9
        self.dropout_prob: float = 0.2
        self.compare_train_test = compare_train_test

    def train(self, parameters: SplitDrugsTestWithTrainTrainingParameterModel) -> TrainingSummaryDTO:

        results = []

        for x_train, x_test, y_train, y_test in super().fold_on_interaction(parameters.drug_data, parameters.interaction_data,
                                                                            train_id=parameters.train_id,
                                                                            padding=True, pca_generating=True):

            model = DeepDDITrainModel(parameters.train_id, self.num_classes, parameters.interaction_data)
            result = model.fit_model(x_train, y_train, x_test, y_test, x_test, y_test)

            results.append(result)

        return super().calculate_fold_results(results)
