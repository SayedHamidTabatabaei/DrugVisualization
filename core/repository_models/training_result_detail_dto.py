from dataclasses import dataclass


@dataclass
class TrainingResultDetailDTO:

    id: int
    training_id: int
    training_name: str
    training_label: int
    f1_score: float
    accuracy: float
    auc: float
    aupr: float
    recall: float
    precision: float
