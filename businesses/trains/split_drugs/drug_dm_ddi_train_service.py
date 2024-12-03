import gc

from businesses.trains.models.dm_ddi_model import DMDDIModel
from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel
from core.models.training_parameter_models.split_drugs_test_with_train_training_parameter_model import SplitDrugsTestWithTrainTrainingParameterModel
from core.repository_models.training_summary_dto import TrainingSummaryDTO

train_model = TrainModel.Drug_DMDDI or TrainModel.Drug_DMDDI_Test


class DrugDMDDITrainService(TrainBaseService):

    def __init__(self, category, compare_train_test: bool = True):
        super().__init__(category)
        self.compare_train_test = compare_train_test

    def train(self, parameters: SplitDrugsTestWithTrainTrainingParameterModel) -> TrainingSummaryDTO:
        results = []

        fold = 1

        for x_train, x_test, y_train, y_test in super().manual_k_fold_train_test_data(parameters.drug_data, parameters.interaction_data,
                                                                                      is_deep_face=True, compare_train_test=self.compare_train_test):
            model = DMDDIModel(parameters.train_id, self.num_classes)
            result = model.fit_model(x_train, y_train, x_test, y_test, x_test, y_test)

            results.append(result)

            del x_train, x_test, y_train, y_test
            gc.collect()

            print(f"Fold {fold} completed. Memory cleared.")
            fold = fold + 1

        return super().calculate_fold_results(results)
