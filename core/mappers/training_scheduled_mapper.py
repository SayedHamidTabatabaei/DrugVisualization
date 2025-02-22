from common.enums.loss_functions import LossFunctions
from common.enums.train_models import TrainModel
from core.repository_models.training_scheduled_dto import TrainingScheduledDTO


def map_training_schedule(query_results) -> TrainingScheduledDTO:
    id, name, description, train_model, loss_function, class_weight, is_test_algorithm, training_conditions, schedule_date, min_sample_count = query_results[0][0]
    return TrainingScheduledDTO(id=id,
                                name=name,
                                description=description,
                                train_model=TrainModel.from_value(train_model),
                                loss_function=LossFunctions.from_value(loss_function) if loss_function is not None else None,
                                class_weight=bool(class_weight),
                                is_test_algorithm=bool(is_test_algorithm),
                                training_conditions=training_conditions,
                                schedule_date=schedule_date,
                                min_sample_count=min_sample_count)


def map_training_schedules(query_results) -> list[TrainingScheduledDTO]:
    training_schedules = []
    for result in query_results:
        id, name, description, train_model, loss_function, class_weight, is_test_algorithm, training_conditions, schedule_date, min_sample_count = result
        training_schedules.append(TrainingScheduledDTO(id=id,
                                                       name=name,
                                                       description=description,
                                                       train_model=TrainModel.from_value(train_model),
                                                       loss_function=LossFunctions.from_value(loss_function) if loss_function is not None else None,
                                                       class_weight=bool(class_weight),
                                                       is_test_algorithm=bool(is_test_algorithm),
                                                       training_conditions=training_conditions,
                                                       schedule_date=schedule_date,
                                                       min_sample_count=min_sample_count))
    return training_schedules
