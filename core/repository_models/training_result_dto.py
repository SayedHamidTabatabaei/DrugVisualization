from dataclasses import dataclass

from common.enums.train_models import TrainModel


@dataclass
class TrainingResultDTO:
    id: int
    name: str
    description: str
    train_model: TrainModel
    training_conditions: str
    f1_score: float
    accuracy: float
    loss: float
    auc: float
    aupr: float
    recall: float
    precision: float
    execute_time: str
