import gc

# noinspection PyUnresolvedReferences
from tensorflow.keras.models import Model

from businesses.trains.models.gat_enc_model import GatEncTrainModel
from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from core.models.training_parameter_models.split_interaction_similarities_training_parameter_model import SplitInteractionSimilaritiesTrainingParameterModel
from core.models.training_params import TrainingParams
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.GAT_Enc_Con_DNN


class GatEncConDnnTrainService(TrainBaseService):

    def __init__(self, category: TrainModel):
        super().__init__(category)
        self.encoding_dim = 128
        self.gat_units = 64
        self.num_heads = 4
        self.dense_units = [512, 256]
        self.droprate = 0.3

    def train(self, parameters: SplitInteractionSimilaritiesTrainingParameterModel) -> (TrainingSummaryDTO, object):

        categories = self.unique_category(parameters.drug_data[0].train_values)

        x_train, x_val, x_test, y_train, y_val, y_test = super().split_deepface_train_val_test(parameters.drug_data, parameters.interaction_data,
                                                                                               train_id=parameters.train_id)

        training_params = TrainingParams(train_id=parameters.train_id, optimizer='adam', loss=parameters.loss_function, class_weight=parameters.class_weight)

        model = GatEncTrainModel(parameters.train_id, categories, self.num_classes, parameters.interaction_data, training_params=training_params)
        result = model.fit_model(x_train, y_train, x_val, y_val, x_test, y_test)

        del x_train, x_test, y_train, y_test
        gc.collect()

        return result
