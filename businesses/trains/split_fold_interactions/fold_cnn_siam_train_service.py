from businesses.trains.models.cnn_siam_model import CNNSiamTrainModel
from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from core.models.training_parameter_models.split_drugs_test_with_train_training_parameter_model import SplitDrugsTestWithTrainTrainingParameterModel
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.Fold_CNN_Siam


class FoldCNNSiamTrainService(TrainBaseService):

    def __init__(self, category):
        super().__init__(category)

    def train(self, parameters: SplitDrugsTestWithTrainTrainingParameterModel) -> TrainingSummaryDTO:
        results = []

        for x_train, x_test, y_train, y_test in super().fold_cnn_on_interaction(parameters.drug_data, parameters.interaction_data,
                                                                                train_id=parameters.train_id):

            model = CNNSiamTrainModel(parameters.train_id, self.num_classes, parameters.interaction_data)
            result = model.fit_model(x_train, y_train, x_test, y_test, x_test, y_test)

            results.append(result)

        return super().calculate_fold_results(results)
