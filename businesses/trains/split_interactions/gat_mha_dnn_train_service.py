import gc

from businesses.trains.models.gat_mha_model import GatMhaTrainModel
from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from core.models.training_parameter_models.split_interaction_similarities_training_parameter_model import SplitInteractionSimilaritiesTrainingParameterModel
from core.models.training_params import TrainingParams
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.GAT_MHA_DNN


class GatMhaDnnTrainService(TrainBaseService):

    def __init__(self, category: TrainModel):
        super().__init__(category)

    def train(self, parameters: SplitInteractionSimilaritiesTrainingParameterModel) -> (TrainingSummaryDTO, object):
        x_train, x_val, x_test, y_train, y_val, y_test = super().split_deepface_train_val_test(parameters.drug_data, parameters.interaction_data,
                                                                                               train_id=parameters.train_id,
                                                                                               mean_of_text_embeddings=False, as_ndarray=False)

        categories = self.unique_category(parameters.drug_data[0].train_values)

        training_params = TrainingParams(train_id=parameters.train_id, optimizer='adam', loss=parameters.loss_function, class_weight=parameters.class_weight)

        model = GatMhaTrainModel(parameters.train_id, categories, self.num_classes, parameters.interaction_data, training_params=training_params)
        result = model.fit_model(x_train, y_train, x_val, y_val, x_test, y_test)

        del x_train, x_test, y_train, y_test
        gc.collect()

        return result
