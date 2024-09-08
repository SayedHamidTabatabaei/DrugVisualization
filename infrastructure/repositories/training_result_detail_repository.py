from core.domain.training_result_detail import TrainingResultDetail
from infrastructure.mysqldb.mysql_repository import MySqlRepository


class TrainingResultDetailRepository(MySqlRepository):
    def __init__(self):
        super().__init__()
        self.table_name = 'training_result_details'

    def insert(self, training_result_id: int, training_label: int, f1_score: float, accuracy: float,
               auc: float, aupr: float) \
            -> TrainingResultDetail:
        training_result_detail = TrainingResultDetail(training_result_id=training_result_id,
                                                      training_label=training_label,
                                                      f1_score=f1_score,
                                                      accuracy=accuracy,
                                                      auc=auc,
                                                      aupr=aupr)

        super().insert(training_result_detail)

        return training_result_detail

    def insert_if_not_exits(self, training_result_id: int, training_label: int, f1_score: float, accuracy: float,
                            auc: float, aupr: float) -> TrainingResultDetail | None:
        is_exists = self.is_exists_training_result_detail(training_result_id, training_label)

        if is_exists:
            return None

        reduction_data = self.insert(training_result_id, training_label, f1_score, accuracy, auc, aupr)

        return reduction_data

    def insert_batch_check_duplicate(self, training_result_details: list[TrainingResultDetail]):
        super().insert_batch_check_duplicate(training_result_details,
                                             [TrainingResultDetail.f1_score,
                                              TrainingResultDetail.accuracy,
                                              TrainingResultDetail.auc,
                                              TrainingResultDetail.aupr])

    def is_exists_training_result_detail(self, training_result_id: int, training_label: int) \
            -> bool:
        result, _ = self.call_procedure('FindTrainingResultDetail',
                                        [training_result_id, training_label])

        training_result_detail = result[0]

        return (training_result_detail is not None and
                (training_result_detail != []
                 if isinstance(training_result_detail, list)
                 else bool(training_result_detail)))
