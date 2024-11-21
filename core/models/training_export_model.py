from dataclasses import dataclass
from datetime import datetime

# from dataclasses_json import dataclass_json

from common.enums.loss_functions import LossFunctions
from common.enums.train_models import TrainModel
from common.enums.training_result_type import TrainingResultType
from core.domain.training_result import TrainingResult
from core.domain.training_result_detail import TrainingResultDetail


# @dataclass_json
@dataclass
class TrainingResultExportModel:
    training_id: int
    training_result_type: TrainingResultType
    result_value: float

    def to_training_result(self, training_id) -> TrainingResult:
        return TrainingResult(training_id=training_id, training_result_type=self.training_result_type, result_value=self.result_value)


# @dataclass_json
@dataclass
class TrainingResultDetailExportModel:
    training_id: int
    training_label: int
    f1_score: float
    accuracy: float
    auc: float
    aupr: float
    recall: float
    precision: float

    def to_training_result_details(self, training_id) -> TrainingResultDetail:
        return TrainingResultDetail(training_id=training_id, training_label=self.training_label, f1_score=self.f1_score, accuracy=self.accuracy, auc=self.auc,
                                    aupr=self.aupr, recall=self.recall, precision=self.precision)


# @dataclass_json
@dataclass
class TrainingExportModel:
    training_id: int

    name: str
    description: str
    train_model: TrainModel
    loss_function: LossFunctions
    class_weight: bool
    is_test_algorithm: bool
    training_conditions: str
    model_parameters: str
    data_report: str
    fold_result_details: str
    execute_time: datetime
    min_sample_count: int

    training_results: list[TrainingResultExportModel]
    training_result_details: list[TrainingResultDetailExportModel]
