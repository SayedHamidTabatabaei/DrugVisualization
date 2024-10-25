from tensorflow.keras.layers import Dense, Input
# noinspection PyUnresolvedReferences
from tensorflow.keras.models import Sequential

from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from core.models.training_parameter_models.split_drugs_test_with_train_training_parameter_model import SplitDrugsTestWithTrainTrainingParameterModel
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.Test


class TrainPlanTest(TrainBaseService):
    def train(self, parameters: SplitDrugsTestWithTrainTrainingParameterModel) -> TrainingSummaryDTO:

        for x_train, x_test, y_train, y_test in super().manual_k_fold_train_test_data(parameters.drug_data, parameters.interaction_data, padding=True):
            pass
