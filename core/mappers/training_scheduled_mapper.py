from common.enums.train_models import TrainModel
from core.repository_models.training_scheduled_dto import TrainingScheduledDTO


def map_training_schedule(query_results) -> TrainingScheduledDTO:
    id, name, description, train_model, is_test_algorithm, training_conditions, schedule_date = query_results[0][0]
    return TrainingScheduledDTO(id=id,
                                name=name,
                                description=description,
                                train_model=TrainModel.from_value(train_model),
                                is_test_algorithm=bool(is_test_algorithm),
                                training_conditions=training_conditions,
                                schedule_date=schedule_date)


def map_training_schedules(query_results) -> list[TrainingScheduledDTO]:
    training_schedules = []
    for result in query_results:
        id, name, description, train_model, is_test_algorithm, training_conditions, schedule_date = result
        training_schedules.append(TrainingScheduledDTO(id=id,
                                                       name=name,
                                                       description=description,
                                                       train_model=TrainModel.from_value(train_model),
                                                       is_test_algorithm=bool(is_test_algorithm),
                                                       training_conditions=training_conditions,
                                                       schedule_date=schedule_date))
    return training_schedules
