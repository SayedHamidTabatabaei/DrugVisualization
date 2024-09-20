from sqlalchemy import Column, Enum as SqlEnum, DateTime, TEXT, NVARCHAR

from common.enums.train_models import TrainModel
from core.domain.base_model import BaseModel


class TrainingScheduled(BaseModel):
    __tablename__ = 'training_scheduled'

    name = Column(NVARCHAR, nullable=True)
    description = Column(NVARCHAR, nullable=True)
    train_model = Column(SqlEnum(TrainModel), nullable=False)
    training_conditions = Column(TEXT, nullable=False)
    schedule_date = Column(DateTime, nullable=False)
