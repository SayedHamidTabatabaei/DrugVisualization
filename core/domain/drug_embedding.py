from sqlalchemy import Column, Integer, String, BOOLEAN, Enum as SqlEnum

from common.enums.embedding_type import EmbeddingType
from common.enums.text_type import TextType
from core.domain.base_model import BaseModel


class DrugEmbedding(BaseModel):
    __tablename__ = 'drug_embeddings'

    drug_id = Column(Integer, nullable=False)
    embedding_type = Column(SqlEnum(EmbeddingType), nullable=False)
    text_type = Column(SqlEnum(TextType), nullable=False)
    embedding = Column(String, nullable=False)
    issue_on_max_length = Column(BOOLEAN, nullable=False)
