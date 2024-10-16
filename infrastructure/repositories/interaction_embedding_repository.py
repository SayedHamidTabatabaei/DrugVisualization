from typing import Union

from common.enums.embedding_type import EmbeddingType
from common.enums.text_type import TextType
from core.domain.interaction_embedding import InteractionEmbedding
from core.mappers.interaction_embedding_mapper import map_interaction_embedding, map_interaction_embedding_dict, map_interaction_text_embedding
from core.repository_models.interaction_embedding_dto import InteractionEmbeddingDTO
from core.repository_models.interaction_text_embedding_dto import InteractionTextEmbeddingDTO
from infrastructure.mysqldb.mysql_repository import MySqlRepository


class InteractionEmbeddingRepository(MySqlRepository):
    def __init__(self):
        super().__init__('interaction_embeddings')

    def insert(self, interaction_id: int, embedding_type: EmbeddingType, text_type: TextType, embedding: str, issue_on_max_length: bool) \
            -> InteractionEmbedding:
        interaction_embedding = InteractionEmbedding(interaction_id=interaction_id,
                                                     embedding_type=embedding_type,
                                                     text_type=text_type,
                                                     embedding=embedding,
                                                     issue_on_max_length=issue_on_max_length)

        super().insert(interaction_embedding)

        return interaction_embedding

    def insert_if_not_exits(self, interaction_id: int, embedding_type: EmbeddingType, text_type: TextType, embedding: str, issue_on_max_length: bool) \
            -> Union[InteractionEmbedding, None]:
        is_exists = self.is_exists_interaction_embedding(interaction_id, embedding_type, text_type)

        if is_exists:
            return None

        interaction_embedding = self.insert(interaction_id=interaction_id, embedding_type=embedding_type, text_type=text_type,
                                            embedding=embedding, issue_on_max_length=issue_on_max_length)

        return interaction_embedding

    def find_interaction_embedding(self, interaction_id: int, embedding_type: EmbeddingType, text_type: TextType) \
            -> InteractionEmbedding:
        result, _ = self.call_procedure('FindInteractionEmbedding',
                                        [interaction_id, embedding_type.value, text_type.value])

        interaction_embedding = result[0]

        return interaction_embedding

    def is_exists_interaction_embedding(self, interaction_id: int, embedding_type: EmbeddingType, text_type: TextType) \
            -> bool:
        result, _ = self.call_procedure('FindInteractionEmbedding',
                                        [interaction_id, embedding_type.value, text_type.value])

        similarity = result[0]

        return similarity is not None and (similarity != [] if isinstance(similarity, list) else bool(similarity))

    def find_all_embedding(self, embedding_type: EmbeddingType, text_type: TextType) \
            -> list[InteractionEmbeddingDTO]:
        result, _ = self.call_procedure('FindAllInteractionEmbedding', [embedding_type.value, text_type.value])

        return map_interaction_embedding(result[0])

    def find_all_embedding_dict(self, embedding_type: EmbeddingType, text_type: TextType) \
            -> dict:
        result, _ = self.call_procedure('FindAllInteractionEmbedding', [embedding_type.value, text_type.value])

        return map_interaction_embedding_dict(result[0])

    def find_interaction_embedding_description(self, embedding_type: EmbeddingType, start: int, length: int) \
            -> list[InteractionTextEmbeddingDTO]:
        result, _ = self.call_procedure('FindInteractionEmbeddingDescription', [embedding_type.value, start, length])

        return map_interaction_text_embedding(result[0])

    def get_embedding_count(self, embedding_type: EmbeddingType, text_type: TextType):
        result, _ = self.call_procedure('GetInteractionEmbeddingCount', [embedding_type.value, text_type.value])

        return result[0]
