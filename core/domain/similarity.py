from sqlalchemy import Column, Integer, Enum as SqlEnum, DECIMAL

from common.enums.category import Category
from common.enums.similarity_type import SimilarityType
from core.domain.base_model import BaseModel


class Similarity(BaseModel):
    __tablename__ = 'similarities'

    similarity_type = Column(SqlEnum(SimilarityType), nullable=False)
    category = Column(SqlEnum(Category), nullable=False)
    drug_1 = Column(Integer, nullable=False)
    drug_2 = Column(Integer, nullable=False)
    value = Column(DECIMAL(10, 8), nullable=False)
