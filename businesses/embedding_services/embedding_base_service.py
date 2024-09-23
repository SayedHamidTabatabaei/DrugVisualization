import re

from common.enums.embedding_type import EmbeddingType
from configs.config import enable_bert_embedding


class EmbeddingBaseService:
    category: EmbeddingType

    def __init__(self, category: EmbeddingType):
        self.enable_bert_embedding = enable_bert_embedding
        self.category = category

    def embed(self, data) -> (str, bool):
        pass

    @staticmethod
    def parse_string(embedding) -> str:
        str_embedding = str(embedding)

        str_embedding = re.sub(r'\[\[\s*', '[[', str_embedding)
        return re.sub(r'\s+', ' ', str_embedding)
