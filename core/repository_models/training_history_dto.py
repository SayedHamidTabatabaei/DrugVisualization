from dataclasses import dataclass

from common.enums.loss_functions import LossFunctions
from common.enums.train_models import TrainModel


@dataclass
class TrainingHistoryDTO:
    id: int
    name: str
    description: str
    train_model: TrainModel
    loss_function: LossFunctions
    is_test_algorithm: bool
    class_weight: bool
    execute_time: str
    min_sample_count: int
    training_results_count: int
    training_result_details_count: int
