from dataclasses import dataclass

from core.repository_models.training_result_detail_summary_dto import TrainingResultDetailSummaryDTO


@dataclass
class TrainingResultSummaryDTO:

    f1_score: float
    accuracy: float
    loss: float
    auc: float
    aupr: float
    training_result_details: list[TrainingResultDetailSummaryDTO]
