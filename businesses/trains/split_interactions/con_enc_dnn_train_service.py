# noinspection PyUnresolvedReferences
from tensorflow.keras.models import Model

from businesses.trains.models.concat_enc_model import ConcatEncModel
from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from core.models.training_parameter_models.split_interaction_similarities_training_parameter_model import SplitInteractionSimilaritiesTrainingParameterModel
from core.models.training_params import TrainingParams
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.Con_AE_DNN


class ConEncDnnTrainService(TrainBaseService):

    def __init__(self, category: TrainModel):
        super().__init__(category)
        self.encoding_dim = 128

    def train(self, parameters: SplitInteractionSimilaritiesTrainingParameterModel) -> TrainingSummaryDTO:

        x_train, x_test, y_train, y_test = super().split_train_test(parameters.drug_data, parameters.interaction_data, train_id=parameters.train_id,
                                                                    padding=True, flat=True)

        training_params = TrainingParams(train_id=parameters.train_id, optimizer='adam', loss=parameters.loss_function, class_weight=parameters.class_weight)

        model = ConcatEncModel(parameters.train_id, self.num_classes, parameters.interaction_data, training_params=training_params)
        result = model.fit_model(x_train, y_train, x_test, y_test, x_test, y_test)

        return result
