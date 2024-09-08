from sqlalchemy import Column, Enum as SqlEnum, Float, DateTime

from common.enums.training_category import TrainingCategory
from core.domain.base_model import BaseModel


class TrainingResult(BaseModel):
    __tablename__ = 'training_results'

    training_category = Column(SqlEnum(TrainingCategory), nullable=False)
    f1_score = Column(Float, nullable=False)
    accuracy = Column(Float, nullable=False)
    auc = Column(Float, nullable=False)
    aupr = Column(Float, nullable=False)
    execute_time = Column(DateTime, nullable=False)
