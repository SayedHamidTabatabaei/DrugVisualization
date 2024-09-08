from datetime import datetime, timezone

from common.enums.training_category import TrainingCategory
from core.domain.training_result import TrainingResult
from infrastructure.mysqldb.mysql_repository import MySqlRepository


class TrainingResultRepository(MySqlRepository):
    def __init__(self):
        super().__init__()
        self.table_name = 'training_results'

    def insert(self, training_category: TrainingCategory, f1_score: float, accuracy: float, auc: float, aupr: float) \
            -> int:
        reduction_data = TrainingResult(training_category=training_category,
                                        f1_score=f1_score,
                                        accuracy=accuracy,
                                        auc=auc,
                                        aupr=aupr,
                                        execute_time=datetime.now(timezone.utc))

        id = super().insert(reduction_data)

        return id
