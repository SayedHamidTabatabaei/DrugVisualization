from dataclasses import dataclass

from common.enums.train_models import TrainModel


@dataclass
class TrainingScheduledDTO:
    id: int
    name: str
    description: str
    train_model: TrainModel
    is_test_algorithm: bool
    training_conditions: str
    schedule_date: str
