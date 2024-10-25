from dataclasses import dataclass

from core.repository_models.training_result_detail_summary_dto import TrainingResultDetailSummaryDTO
from core.repository_models.training_result_summary_dto import TrainingResultSummaryDTO


@dataclass
class TrainingSummaryDTO:

    training_results: list[TrainingResultSummaryDTO]
    model: bytes | list[bytes]
    data_report: object
    model_info: object
    fold_result_details: object
    training_result_details: list[TrainingResultDetailSummaryDTO]
