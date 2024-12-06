from dataclasses import dataclass

from core.repository_models.training_result_detail_summary_dto import TrainingResultDetailSummaryDTO
from core.repository_models.training_result_summary_dto import TrainingResultSummaryDTO
from typing import Union, List


@dataclass
class TrainingSummaryDTO:

    training_results: list[TrainingResultSummaryDTO]
    model: Union[bytes, List[bytes]]
    data_report: object
    model_info: object
    fold_result_details: object
    training_result_details: list[TrainingResultDetailSummaryDTO]
    incorrect_predictions: dict
