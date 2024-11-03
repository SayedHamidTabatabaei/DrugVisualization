from dataclasses import dataclass

from common.enums.loss_functions import LossFunctions
from common.enums.train_models import TrainModel


@dataclass
class TrainingScheduledDTO:
    id: int
    name: str
    description: str
    train_model: TrainModel
    loss_function: LossFunctions
    class_weight: bool
    is_test_algorithm: bool
    training_conditions: str
    schedule_date: str
    min_sample_count: int
