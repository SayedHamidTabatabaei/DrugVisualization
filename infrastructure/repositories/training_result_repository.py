from datetime import datetime, timezone

from common.enums.train_models import TrainModel
from core.domain.training_result import TrainingResult
from core.mappers import training_result_mapper
from core.repository_models.training_result_dto import TrainingResultDTO
from infrastructure.mysqldb.mysql_repository import MySqlRepository


class TrainingResultRepository(MySqlRepository):
    def __init__(self):
        super().__init__()
        self.table_name = 'training_results'

    def insert(self, train_model: TrainModel, training_conditions: str, f1_score: float, accuracy: float, loss: float,
               auc: float, aupr: float) \
            -> int:
        reduction_data = TrainingResult(train_model=train_model,
                                        training_conditions=training_conditions,
                                        f1_score=f1_score,
                                        accuracy=accuracy,
                                        loss=loss,
                                        auc=auc,
                                        aupr=aupr,
                                        execute_time=datetime.now(timezone.utc))

        id = super().insert(reduction_data)

        return id

    def insert(self, train_model: TrainModel, training_conditions: str) \
            -> int:
        reduction_data = TrainingResult(train_model=train_model,
                                        training_conditions=training_conditions,
                                        f1_score=0.0,
                                        accuracy=0.0,
                                        loss=0.0,
                                        auc=0.0,
                                        aupr=0.0,
                                        execute_time=datetime.now(timezone.utc))

        id = super().insert(reduction_data)

        return id

    def update(self, id: int, f1_score: float, accuracy: float, loss: float, auc: float, aupr: float):

        training_result = self.get_training_result_by_id(id)

        training_result.f1_score = f1_score
        training_result.accuracy = accuracy
        training_result.loss = loss
        training_result.auc = auc
        training_result.aupr = aupr

        update_columns = ['f1_score', 'accuracy', 'loss', 'auc', 'aupr']

        rowcount = super().update(training_result, update_columns)

        return rowcount

    def get_training_result_by_id(self, id) -> TrainingResult:
        result, _ = self.call_procedure('GetTrainingResultById', [id])
        return training_result_mapper.map_training_result(result)

    def get_training_result_count(self, train_model: TrainModel):

        model_value = train_model.value if train_model is not None else None

        result, _ = self.call_procedure('GetTrainingResultCount', [model_value])

        return result[0][0]

    def find_all_training_result(self, train_model: TrainModel, start: int, length: int) \
            -> list[TrainingResultDTO]:

        model_value = train_model.value if train_model is not None else None

        result, _ = self.call_procedure('FindAllTrainingResults',
                                        [model_value, start, length])

        training_result = result[0]

        return training_result_mapper.map_training_results(training_result)
