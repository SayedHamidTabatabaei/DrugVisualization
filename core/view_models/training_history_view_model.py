from dataclasses import dataclass


@dataclass
class TrainingHistoryViewModel:
    id: int
    name: str
    description: str
    train_model: str
    training_conditions: str
    f1_score: float
    accuracy: float
    loss: float
    auc: float
    aupr: float
    recall: float
    precision: float
    execute_time: str
