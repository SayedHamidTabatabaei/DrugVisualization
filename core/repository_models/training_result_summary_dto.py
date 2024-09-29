from dataclasses import dataclass

from common.enums.training_result_type import TrainingResultType


@dataclass
class TrainingResultSummaryDTO:

    training_result_type: TrainingResultType
    result_value: float
