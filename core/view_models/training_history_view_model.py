from dataclasses import dataclass


@dataclass
class TrainingHistoryViewModel:
    id: int
    train_model: str
    training_conditions: str
    f1_score: float
    accuracy: float
    loss: float
    auc: float
    aupr: float
    execute_time: str
