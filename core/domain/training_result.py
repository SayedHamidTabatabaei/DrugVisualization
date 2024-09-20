from sqlalchemy import Column, Enum as SqlEnum, Float, DateTime, TEXT, LargeBinary, NVARCHAR

from common.enums.train_models import TrainModel
from core.domain.base_model import BaseModel


class TrainingResult(BaseModel):
    __tablename__ = 'training_results'

    name = Column(NVARCHAR, nullable=True)
    description = Column(NVARCHAR, nullable=True)
    train_model = Column(SqlEnum(TrainModel), nullable=False)
    training_conditions = Column(TEXT, nullable=False)
    f1_score = Column(Float, nullable=False)
    accuracy = Column(Float, nullable=False)
    loss = Column(Float, nullable=False)
    auc = Column(Float, nullable=False)
    aupr = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    execute_time = Column(DateTime, nullable=False)
