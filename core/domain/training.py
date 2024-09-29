from sqlalchemy import Column, Enum as SqlEnum, DateTime, TEXT, NVARCHAR, BOOLEAN

from common.enums.train_models import TrainModel
from core.domain.base_model import BaseModel


class Training(BaseModel):
    __tablename__ = 'trainings'

    name = Column(NVARCHAR, nullable=True)
    description = Column(NVARCHAR, nullable=True)
    train_model = Column(SqlEnum(TrainModel), nullable=False)
    is_test_algorithm = Column(BOOLEAN, nullable=False)
    training_conditions = Column(TEXT, nullable=False)
    data_report = Column(TEXT, nullable=False)
    execute_time = Column(DateTime, nullable=False)
