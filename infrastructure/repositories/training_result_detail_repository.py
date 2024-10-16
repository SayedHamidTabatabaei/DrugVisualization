import json
from typing import Union

from core.domain.training_result_detail import TrainingResultDetail
from core.mappers import training_mapper
from core.repository_models.training_result_detail_dto import TrainingResultDetailDTO
from infrastructure.mysqldb.mysql_repository import MySqlRepository


class TrainingResultDetailRepository(MySqlRepository):
    def __init__(self):
        super().__init__('training_result_details')

    def insert(self, training_id: int, training_label: int, f1_score: float, accuracy: float,
               auc: float, aupr: float, recall: float, precision: float) \
            -> TrainingResultDetail:
        training_result_detail = TrainingResultDetail(training_id=training_id,
                                                      training_label=training_label,
                                                      f1_score=f1_score,
                                                      accuracy=accuracy,
                                                      auc=auc,
                                                      aupr=aupr,
                                                      recall=recall,
                                                      precision=precision)

        super().insert(training_result_detail)

        return training_result_detail

    def insert_if_not_exits(self, training_id: int, training_label: int, f1_score: float, accuracy: float,
                            auc: float, aupr: float, recall: float, precision: float) \
            -> Union[TrainingResultDetail, None]:
        is_exists = self.is_exists_training_result_detail(training_id, training_label)

        if is_exists:
            return None

        reduction_data = self.insert(training_id, training_label, f1_score, accuracy, auc, aupr, recall,
                                     precision)

        return reduction_data

    def insert_batch_check_duplicate(self, training_result_details: list[TrainingResultDetail]):
        super().insert_batch_check_duplicate(training_result_details,
                                             [TrainingResultDetail.f1_score,
                                              TrainingResultDetail.accuracy,
                                              TrainingResultDetail.auc,
                                              TrainingResultDetail.aupr,
                                              TrainingResultDetail.recall,
                                              TrainingResultDetail.precision])

    def is_exists_training_result_detail(self, training_id: int, training_label: int) \
            -> bool:
        result, _ = self.call_procedure('FindTrainingResultDetail',
                                        [training_id, training_label])

        training_result_detail = result[0]

        return (training_result_detail is not None and
                (training_result_detail != []
                 if isinstance(training_result_detail, list)
                 else bool(training_result_detail)))

    def find_all_training_result_details(self, train_id: int) -> list[TrainingResultDetailDTO]:

        result, _ = self.call_procedure('FindAllTrainingResultDetails', [train_id])

        training_result = result[0]

        return training_mapper.map_training_result_details(training_result)

    def find_training_result_details_by_training_ids(self, train_ids: list[int]) -> list[TrainingResultDetailDTO]:
        ids_json = json.dumps(train_ids)

        result, _ = self.call_procedure('FindTrainingResultDetailsByTrainingIds', [ids_json])

        training_result = result[0]

        return training_mapper.map_training_result_details(training_result)
