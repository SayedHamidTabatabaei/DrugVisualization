from datetime import datetime

from common.enums.loss_functions import LossFunctions
from common.enums.train_models import TrainModel
from core.domain.training_scheduled import TrainingScheduled
from core.mappers import training_scheduled_mapper
from core.repository_models.training_scheduled_dto import TrainingScheduledDTO
from infrastructure.mysqldb.mysql_repository import MySqlRepository


class TrainingScheduledRepository(MySqlRepository):
    def __init__(self):
        super().__init__('training_scheduled')

    def insert(self, name: str, description: str, train_model: TrainModel, loss_function: LossFunctions, class_weight: bool, is_test_algorithm: bool,
               training_conditions: str, schedule_date: datetime, min_sample_count: int) \
            -> int:
        data = TrainingScheduled(name=name,
                                 description=description,
                                 train_model=train_model,
                                 loss_function=loss_function,
                                 class_weight=class_weight,
                                 is_test_algorithm=bool(is_test_algorithm),
                                 training_conditions=training_conditions,
                                 schedule_date=schedule_date,
                                 min_sample_count=min_sample_count)

        id = super().insert(data)

        return id

    def delete(self, id: int):

        rowcount = super().delete(id)

        return rowcount

    def get_training_scheduled_by_id(self, id) -> TrainingScheduledDTO:
        result, _ = self.call_procedure('GetTrainingScheduledById', [id])
        return training_scheduled_mapper.map_training_schedule(result)

    def find_overdue_training_scheduled(self) \
            -> list[TrainingScheduledDTO]:
        result, _ = self.call_procedure('FindOverdueTrainingScheduled')

        training_schedules = result[0]

        return training_scheduled_mapper.map_training_schedules(training_schedules)

    def find_all_training_scheduled(self, train_model: TrainModel, start: int, length: int) \
            -> list[TrainingScheduledDTO]:
        model_value = train_model.value if train_model is not None else None

        result, _ = self.call_procedure('FindAllTrainingScheduled',
                                        [model_value, start, length])

        training_schedules = result[0]

        return training_scheduled_mapper.map_training_schedules(training_schedules)
