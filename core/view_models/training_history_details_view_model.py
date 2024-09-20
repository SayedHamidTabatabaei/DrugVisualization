from dataclasses import dataclass


@dataclass
class TrainingHistoryDetailsViewModel:
    training_label: int
    f1_score: float
    accuracy: float
    auc: float
    aupr: float
    recall: float
    precision: float
