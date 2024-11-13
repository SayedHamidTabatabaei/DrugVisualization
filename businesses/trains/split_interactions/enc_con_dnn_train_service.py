from businesses.trains.models.enc_concat_model import EncConcatModel
from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from core.models.training_parameter_models.split_interaction_similarities_training_parameter_model import SplitInteractionSimilaritiesTrainingParameterModel
from core.models.training_params import TrainingParams
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.Enc_Con_DNN


class EncConDnnTrainService(TrainBaseService):

    def __init__(self, category: TrainModel):
        super().__init__(category)

    def train(self, parameters: SplitInteractionSimilaritiesTrainingParameterModel) -> (TrainingSummaryDTO, object):

        x_train, x_test, y_train, y_test = super().split_train_test(parameters.drug_data, parameters.interaction_data, train_id=parameters.train_id,
                                                                    padding=True)

        training_params = TrainingParams(train_id=parameters.train_id, optimizer='adam', loss=parameters.loss_function, class_weight=parameters.class_weight)

        model = EncConcatModel(parameters.train_id, self.num_classes, parameters.interaction_data, training_params=training_params)
        result = model.fit_model(x_train, y_train, x_test, y_test, x_test, y_test)

        return result
