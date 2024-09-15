from dataclasses import dataclass


@dataclass
class TrainingResultDetailSummaryDTO:

    training_label: int
    f1_score: float
    accuracy: float
    auc: float
    aupr: float
