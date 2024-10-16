from dataclasses import dataclass

from common.enums.training_result_type import TrainingResultType


@dataclass
class TrainingResultReportDTO:
    id: int
    training_id: int
    training_name: str
    training_result_type: TrainingResultType
    result_value: float
