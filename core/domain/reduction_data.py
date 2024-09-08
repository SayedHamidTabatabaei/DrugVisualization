from sqlalchemy import Column, Integer, Enum as SqlEnum, BOOLEAN, Text

from common.enums.category import Category
from common.enums.reduction_category import ReductionCategory
from common.enums.similarity_type import SimilarityType
from core.domain.base_model import BaseModel


class ReductionData(BaseModel):
    __tablename__ = 'reduction_data'

    drug_id = Column(Integer, nullable=False)
    similarity_type = Column(SqlEnum(SimilarityType), nullable=False)
    category = Column(SqlEnum(Category), nullable=False)
    reduction_category = Column(SqlEnum(ReductionCategory), nullable=False)
    reduction_values = Column(Text, nullable=False)
    has_enzyme = Column(BOOLEAN, nullable=False)
    has_pathway = Column(BOOLEAN, nullable=False)
    has_target = Column(BOOLEAN, nullable=False)
    has_smiles = Column(BOOLEAN, nullable=False)
