from sqlalchemy import Column, Enum as SqlEnum, INT, Float

from common.enums.training_result_type import TrainingResultType
from core.domain.base_model import BaseModel


class TrainingResult(BaseModel):
    __tablename__ = 'training_results'

    training_id = Column(INT, nullable=False)
    training_result_type = Column(SqlEnum(TrainingResultType), nullable=False)
    result_value = Column(Float, nullable=False)
