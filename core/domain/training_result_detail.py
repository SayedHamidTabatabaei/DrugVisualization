from sqlalchemy import Column, Float, Integer

from core.domain.base_model import BaseModel


class TrainingResultDetail(BaseModel):
    __tablename__ = 'training_result_details'

    training_id = Column(Integer, nullable=False)
    training_label = Column(Integer, nullable=False)
    f1_score = Column(Float, nullable=False)
    accuracy = Column(Float, nullable=False)
    auc = Column(Float, nullable=False)
    aupr = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
