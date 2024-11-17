from businesses.trains.models.cnn_ddi_model import CNNDDITrainModel
from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from core.models.training_parameter_models.split_interaction_similarities_training_parameter_model import SplitInteractionSimilaritiesTrainingParameterModel
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.CNN_DDI


class CNNDDITrainService(TrainBaseService):

    def __init__(self, category):
        super().__init__(category)

    def train(self, parameters: SplitInteractionSimilaritiesTrainingParameterModel) -> TrainingSummaryDTO:

        x_train, x_test, x_val, y_train, y_val, y_test = super().split_cnn_train_test(parameters.drug_data, parameters.interaction_data,
                                                                                      train_id=parameters.train_id)

        model = CNNDDITrainModel(parameters.train_id, self.num_classes, parameters.interaction_data)
        result = model.fit_model(x_train, y_train, x_val, y_val, x_test, y_test)

        return result
