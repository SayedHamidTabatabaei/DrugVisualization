from typing import Union

from common.enums.training_result_type import TrainingResultType
from core.domain.training_result import TrainingResult
from core.mappers import training_mapper
from core.repository_models.training_result_detail_dto import TrainingResultDetailDTO
from core.repository_models.training_result_report_dto import TrainingResultReportDTO
from infrastructure.mysqldb.mysql_repository import MySqlRepository


class TrainingResultRepository(MySqlRepository):
    def __init__(self):
        super().__init__('training_results')

    def insert(self, training_id: int, training_result_type: int, result_value: float) \
            -> TrainingResult:
        training_result_detail = TrainingResult(training_id=training_id,
                                                training_result_type=training_result_type,
                                                result_value=result_value)

        super().insert(training_result_detail)

        return training_result_detail

    def insert_if_not_exits(self, training_id: int, training_result_type: int, result_value: float) \
            -> Union[TrainingResult, None]:
        is_exists = self.is_exists_training_result(training_id, training_result_type)

        if is_exists:
            return None

        data = self.insert(training_id, training_result_type, result_value)

        return data

    def insert_batch_check_duplicate(self, training_results: list[TrainingResult]):
        super().insert_batch_check_duplicate(training_results, [TrainingResult.result_value])

    def is_exists_training_result(self, training_id: int, training_result_type: int) \
            -> bool:
        result, _ = self.call_procedure('FindTrainingResult', [training_id, training_result_type])

        training_result = result[0]

        return (training_result is not None and
                (training_result != []
                 if isinstance(training_result, list)
                 else bool(training_result)))

    def find_all_training_results(self, train_id: int) -> list[TrainingResult]:
        result, _ = self.call_procedure('FindAllTrainingResults', [train_id])

        training_result = result[0]

        return training_mapper.map_training_results(training_result)

    def get_training_result(self, train_id: int, training_result_type: TrainingResultType) -> TrainingResultReportDTO:
        result, _ = self.call_procedure('FindTrainingResult', [train_id, training_result_type.value])

        training_result = result[0][0]

        return training_mapper.map_training_result_report_dto(training_result)
