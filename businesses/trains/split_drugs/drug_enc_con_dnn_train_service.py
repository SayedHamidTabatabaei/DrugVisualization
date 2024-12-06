# noinspection PyUnresolvedReferences
from tensorflow.keras.models import Model

from businesses.trains.models.enc_concat_model import EncConcatModel
from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from core.models.training_parameter_models.split_drugs_test_with_train_training_parameter_model import SplitDrugsTestWithTrainTrainingParameterModel
from core.models.training_params import TrainingParams
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.Drug_Enc_Con_DNN or TrainModel.Drug_Enc_Con_DNN_Test


class DrugEncConDnnTrainService(TrainBaseService):

    def __init__(self, category: TrainModel, compare_train_test: bool = True, file_train_id: int = None):
        super().__init__(category, file_train_id=file_train_id)
        self.compare_train_test = compare_train_test

    def train(self, parameters: SplitDrugsTestWithTrainTrainingParameterModel) -> TrainingSummaryDTO:

        training_params = TrainingParams(train_id=parameters.train_id, optimizer='adam', loss=parameters.loss_function, class_weight=parameters.class_weight)

        results = []

        for x_train, x_test, y_train, y_test in super().manual_k_fold_train_test_data(parameters.drug_data, parameters.interaction_data,
                                                                                      padding=True, compare_train_test=self.compare_train_test):

            model = EncConcatModel(parameters.train_id, self.num_classes, parameters.interaction_data, training_params=training_params)
            result = model.fit_model(x_train, y_train, x_test, y_test, x_test, y_test)

            results.append(result)

        return super().calculate_fold_results(results)

