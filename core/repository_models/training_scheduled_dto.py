from dataclasses import dataclass

from common.enums.train_models import TrainModel


@dataclass
class TrainingScheduledDTO:
    id: int
    name: str
    description: str
    train_model: TrainModel
    training_conditions: str
    schedule_date: str
