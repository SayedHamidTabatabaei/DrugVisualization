from datetime import datetime, timezone

from common.enums.train_models import TrainModel
from core.domain.training import Training
from core.mappers import training_mapper
from core.repository_models.training_result_dto import TrainingResultDTO
from infrastructure.mysqldb.mysql_repository import MySqlRepository


class TrainingRepository(MySqlRepository):
    def __init__(self):
        super().__init__('trainings')

    def insert(self, name: str, description: str, train_model: TrainModel, is_test_algorithm: bool, training_conditions: str) -> int:
        data = Training(name=name,
                        description=description,
                        train_model=train_model,
                        is_test_algorithm=bool(is_test_algorithm),
                        training_conditions=training_conditions,
                        data_report="",
                        execute_time=datetime.now(timezone.utc))

        id = super().insert(data)

        return id

    def update(self, id: int, data_report: str) -> None:

        training = self.get_training_by_id(id)

        training.data_report = data_report

        update_columns = ['data_report']

        super().update(training, update_columns)

    def get_training_by_id(self, id) -> Training:
        result, _ = self.call_procedure('GetTrainingById', [id])
        return training_mapper.map_training(result)

    def get_training_count(self, train_model: TrainModel):
        model_value = train_model.value if train_model is not None else None

        result, _ = self.call_procedure('GetTrainingCount', [model_value])

        return result[0][0]

    def find_all_training(self, train_model: TrainModel, start: int, length: int) \
            -> list[TrainingResultDTO]:
        model_value = train_model.value if train_model is not None else None

        result, _ = self.call_procedure('FindAllTrainings',
                                        [model_value, start, length])

        return training_mapper.map_trainings(result[0])
