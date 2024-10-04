from sqlalchemy import Column, Enum as SqlEnum, DateTime, TEXT, NVARCHAR, BOOLEAN

from common.enums.loss_functions import LossFunctions
from common.enums.train_models import TrainModel
from core.domain.base_model import BaseModel


class TrainingScheduled(BaseModel):
    __tablename__ = 'training_scheduled'

    name = Column(NVARCHAR, nullable=True)
    description = Column(NVARCHAR, nullable=True)
    train_model = Column(SqlEnum(TrainModel), nullable=False)
    loss_function = Column(SqlEnum(LossFunctions), nullable=True)
    is_test_algorithm = Column(BOOLEAN, nullable=False)
    class_weight = Column(BOOLEAN, nullable=False)
    training_conditions = Column(TEXT, nullable=False)
    schedule_date = Column(DateTime, nullable=False)
