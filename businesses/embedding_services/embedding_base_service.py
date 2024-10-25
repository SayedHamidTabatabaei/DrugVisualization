import re
import numpy as np

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
        embedding = embedding.detach().numpy()

        str_embedding = np.array2string(embedding, separator=', ', threshold=np.inf)

        str_embedding = re.sub(r'\n\s*', ' ', str_embedding)
        str_embedding = re.sub(r'\s+', ' ', str_embedding)

        return str_embedding
