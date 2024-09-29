from dataclasses import dataclass

from core.repository_models.training_result_detail_summary_dto import TrainingResultDetailSummaryDTO
from core.repository_models.training_result_summary_dto import TrainingResultSummaryDTO


@dataclass
class TrainingSummaryDTO:

    training_results: list[TrainingResultSummaryDTO]
    model: bytes
    data_report: object
    training_result_details: list[TrainingResultDetailSummaryDTO]
