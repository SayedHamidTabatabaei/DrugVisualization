from common.enums.train_models import TrainModel
from core.domain.training_result import TrainingResult
from core.repository_models.training_result_detail_dto import TrainingResultDetailDTO
from core.repository_models.training_result_dto import TrainingResultDTO


def map_training_result(query_results) -> TrainingResult:
    id, train_model, training_conditions, f1_score, accuracy, loss, auc, aupr, execute_time = query_results[0][0]
    return TrainingResult(id=id,
                          train_model=train_model,
                          training_conditions=training_conditions,
                          f1_score=f1_score,
                          accuracy=accuracy,
                          loss=loss,
                          auc=auc,
                          aupr=aupr,
                          execute_time=execute_time)


def map_training_results(query_results) -> list[TrainingResultDTO]:
    training_results = []
    for result in query_results:
        id, train_model, training_conditions, f1_score, accuracy, loss, auc, aupr, execute_time = result

        training_results.append(TrainingResultDTO(id=id,
                                                  train_model=TrainModel(train_model),
                                                  training_conditions=training_conditions,
                                                  f1_score=f1_score,
                                                  accuracy=accuracy,
                                                  loss=loss,
                                                  auc=auc,
                                                  aupr=aupr,
                                                  execute_time=execute_time))
    return training_results


def map_training_result_details(query_results) -> list[TrainingResultDetailDTO]:
    training_result_details = []
    for result in query_results:
        id, training_result_id, training_label, f1_score, accuracy, auc, aupr = result

        training_result_details.append(TrainingResultDetailDTO(id=id,
                                                               training_result_id=training_result_id,
                                                               training_label=training_label,
                                                               f1_score=f1_score,
                                                               accuracy=accuracy,
                                                               auc=auc,
                                                               aupr=aupr))
    return training_result_details
