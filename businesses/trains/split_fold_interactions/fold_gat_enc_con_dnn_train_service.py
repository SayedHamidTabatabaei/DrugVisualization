import gc

from businesses.trains.models.gat_enc_model import GatEncTrainModel
from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from core.models.hyper_params import HyperParams
from core.models.training_parameter_models.split_drugs_test_with_train_training_parameter_model import SplitDrugsTestWithTrainTrainingParameterModel
from core.models.training_params import TrainingParams
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.Fold_GAT_Enc_Con_DNN


class FoldGatEncConDnnTrainService(TrainBaseService):

    def __init__(self, category, encoding_dim, gat_units, num_heads, dense_units, droprate, pooling_mode = None):
        super().__init__(category)
        self.hyper_params = HyperParams(encoding_dim, gat_units, num_heads, dense_units, droprate, pooling_mode)

    def train(self, parameters: SplitDrugsTestWithTrainTrainingParameterModel) -> TrainingSummaryDTO:

        categories = self.unique_category(parameters.drug_data[0].train_values)

        training_params = TrainingParams(train_id=parameters.train_id, optimizer='adam', loss=parameters.loss_function, class_weight=parameters.class_weight)

        results = []

        fold = 1

        for x_train, x_test, y_train, y_test in super().fold_on_interaction_deepface(parameters.drug_data, parameters.interaction_data,
                                                                                     train_id=parameters.train_id):

            model = GatEncTrainModel(parameters.train_id, categories, self.num_classes, parameters.interaction_data, training_params=training_params, hyper_params=self.hyper_params)
            result = model.fit_model(x_train, y_train, x_test, y_test, x_test, y_test)

            results.append(result)

            del x_train, x_test, y_train, y_test
            gc.collect()

            print(f"Fold {fold} completed. Memory cleared.")
            fold = fold + 1

        return super().calculate_fold_results(results)
