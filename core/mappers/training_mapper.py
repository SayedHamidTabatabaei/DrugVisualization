from common.enums.loss_functions import LossFunctions
from common.enums.train_models import TrainModel
from core.domain.training import Training
from core.domain.training_result import TrainingResult
from core.models.training_export_model import TrainingExportModel, TrainingResultExportModel, TrainingResultDetailExportModel
from core.repository_models.training_history_dto import TrainingHistoryDTO
from core.repository_models.training_result_detail_dto import TrainingResultDetailDTO
from core.repository_models.training_result_dto import TrainingResultDTO
from core.repository_models.training_result_report_dto import TrainingResultReportDTO


def map_training(query_results) -> Training:
    (id, name, description, train_model, loss_function, class_weight, is_test_algorithm, training_conditions, model_parameters, data_report,
     fold_result_details, execute_time, min_sample_count) = query_results[0][0]
    return Training(id=id,
                    name=name,
                    description=description,
                    train_model=train_model,
                    loss_function=LossFunctions.from_value(loss_function) if loss_function is not None else None,
                    is_test_algorithm=bool(is_test_algorithm),
                    class_weight=bool(class_weight),
                    training_conditions=training_conditions,
                    model_parameters=model_parameters,
                    data_report=data_report,
                    fold_result_details=fold_result_details,
                    execute_time=execute_time,
                    min_sample_count=min_sample_count)


def map_training_list(query_results) -> list[Training]:
    training_results = []
    for result in query_results[0]:
        (id, name, description, train_model, loss_function, class_weight, is_test_algorithm, training_conditions, model_parameters, data_report,
         fold_result_details, execute_time, min_sample_count) = result
        training_results.append(Training(id=id,
                                         name=name,
                                         description=description,
                                         train_model=train_model,
                                         loss_function=LossFunctions.from_value(loss_function) if loss_function is not None else None,
                                         is_test_algorithm=bool(is_test_algorithm),
                                         class_weight=bool(class_weight),
                                         training_conditions=training_conditions,
                                         model_parameters=model_parameters,
                                         data_report=data_report,
                                         fold_result_details=fold_result_details,
                                         execute_time=execute_time,
                                         min_sample_count=min_sample_count))

    return training_results


def map_trainings(query_results) -> list[TrainingResultDTO]:
    training_results = []
    for result in query_results:
        (id, name, description, train_model, loss_function, class_weight, is_test_algorithm, training_conditions, model_parameters, data_report,
         execute_time, min_sample_count,
         accuracy, loss, f1_score_weighted, f1_score_micro, f1_score_macro,
         auc_weighted, auc_micro, auc_macro,
         aupr_weighted, aupr_micro, aupr_macro,
         recall_weighted, recall_micro, recall_macro,
         precision_weighted, precision_micro, precision_macro) = result

        training_results.append(TrainingResultDTO(id=id,
                                                  name=name,
                                                  description=description,
                                                  train_model=TrainModel.from_value(train_model),
                                                  loss_function=LossFunctions.from_value(loss_function) if loss_function is not None else None,
                                                  is_test_algorithm=bool(is_test_algorithm),
                                                  class_weight=bool(class_weight),
                                                  training_conditions=training_conditions,
                                                  model_parameters=model_parameters,
                                                  accuracy=accuracy,
                                                  loss=loss,
                                                  f1_score_weighted=f1_score_weighted,
                                                  f1_score_micro=f1_score_micro,
                                                  f1_score_macro=f1_score_macro,
                                                  auc_weighted=auc_weighted,
                                                  auc_micro=auc_micro,
                                                  auc_macro=auc_macro,
                                                  aupr_weighted=aupr_weighted,
                                                  aupr_micro=aupr_micro,
                                                  aupr_macro=aupr_macro,
                                                  recall_weighted=recall_weighted,
                                                  recall_micro=recall_micro,
                                                  recall_macro=recall_macro,
                                                  precision_weighted=precision_weighted,
                                                  precision_micro=precision_micro,
                                                  precision_macro=precision_macro,
                                                  execute_time=execute_time,
                                                  min_sample_count=min_sample_count))
    return training_results


def map_training_histories(query_results) -> list[TrainingHistoryDTO]:
    training_results = []
    for result in query_results:
        (id, name, description, train_model, loss_function, class_weight, is_test_algorithm, execute_time, min_sample_count, training_results_count,
         training_result_details_count) = result

        training_results.append(TrainingHistoryDTO(id=id,
                                                   name=name,
                                                   description=description,
                                                   train_model=TrainModel.from_value(train_model),
                                                   loss_function=LossFunctions.from_value(loss_function) if loss_function is not None else None,
                                                   is_test_algorithm=bool(is_test_algorithm),
                                                   class_weight=bool(class_weight),
                                                   execute_time=execute_time,
                                                   min_sample_count=min_sample_count,
                                                   training_results_count=training_results_count,
                                                   training_result_details_count=training_result_details_count))
    return training_results


def map_training_results(query_results):
    training_results = []
    for result in query_results:
        id, training_id, training_result_type, result_value = result

        training_results.append(TrainingResult(id=id,
                                               training_result_type=training_result_type,
                                               result_value=result_value))
    return training_results


def map_training_result(query_results):
    id, training_id, name, training_result_type, result_value = query_results

    return TrainingResult(id=id, training_id=training_id, training_result_type=training_result_type, result_value=result_value)


def map_training_result_report_dto(query_results):
    id, training_id, name, training_result_type, result_value = query_results

    return TrainingResultReportDTO(id=id, training_id=training_id, training_name=name, training_result_type=training_result_type, result_value=result_value)


def map_training_result_details(query_results) -> list[TrainingResultDetailDTO]:
    training_result_details = []
    for result in query_results:
        id, training_id, training_name, training_label, f1_score, accuracy, auc, aupr, recall, precision = result

        training_result_details.append(TrainingResultDetailDTO(id=id,
                                                               training_id=training_id,
                                                               training_name=training_name,
                                                               training_label=training_label,
                                                               f1_score=f1_score,
                                                               accuracy=accuracy,
                                                               auc=auc,
                                                               aupr=aupr,
                                                               recall=recall,
                                                               precision=precision))
    return training_result_details


def map_training_export(training: Training, training_results: list[TrainingResult], training_result_details: list[TrainingResultDetailDTO]) \
        -> TrainingExportModel:
    return TrainingExportModel(training_id=training.id,
                               name=training.name,
                               description=training.description,
                               train_model=training.train_model,
                               loss_function=training.loss_function,
                               class_weight=training.class_weight,
                               is_test_algorithm=training.is_test_algorithm,
                               training_conditions=training.training_conditions,
                               model_parameters=training.model_parameters,
                               data_report=training.data_report,
                               fold_result_details=training.fold_result_details,
                               execute_time=training.execute_time,
                               min_sample_count=training.min_sample_count,
                               training_results=[TrainingResultExportModel(training_id=tr.training_id,
                                                                           training_result_type=tr.training_result_type,
                                                                           result_value=tr.result_value)
                                                 for tr in training_results],
                               training_result_details=[TrainingResultDetailExportModel(training_id=trd.training_id,
                                                                                        training_label=trd.training_label,
                                                                                        f1_score=trd.f1_score,
                                                                                        accuracy=trd.accuracy,
                                                                                        auc=trd.auc,
                                                                                        aupr=trd.aupr,
                                                                                        recall=trd.recall,
                                                                                        precision=trd.precision)
                                                        for trd in training_result_details]
                               )
