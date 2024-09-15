from dataclasses import dataclass


@dataclass
class TrainingResultDetailDTO:

    id: int
    training_result_id: int
    training_label: int
    f1_score: float
    accuracy: float
    auc: float
    aupr: float
