from typing import Union

from common.enums.embedding_type import EmbeddingType
from common.enums.text_type import TextType
from core.domain.drug_embedding import DrugEmbedding
from core.mappers.drug_embedding_mapper import map_drug_embedding, map_text_embedding, map_drug_embedding_dict
from core.repository_models.drug_embedding_dto import DrugEmbeddingDTO
from core.repository_models.text_embedding_dto import TextEmbeddingDTO
from infrastructure.mysqldb.mysql_repository import MySqlRepository


class DrugEmbeddingRepository(MySqlRepository):
    def __init__(self):
        super().__init__('drug_embeddings')

    def insert(self, drug_id: int, embedding_type: EmbeddingType, text_type: TextType, embedding: str, issue_on_max_length: bool) \
            -> DrugEmbedding:
        drug_embedding = DrugEmbedding(drug_id=drug_id,
                                       embedding_type=embedding_type,
                                       text_type=text_type,
                                       embedding=embedding,
                                       issue_on_max_length=issue_on_max_length)

        super().insert(drug_embedding)

        return drug_embedding

    def insert_if_not_exits(self, drug_id: int, embedding_type: EmbeddingType, text_type: TextType, embedding: str) \
            -> Union[DrugEmbedding, None]:

        is_exists = self.is_exists_drug_embedding(drug_id, embedding_type, text_type)

        if is_exists:
            return None

        drug_embedding = self.insert(drug_id=drug_id, embedding_type=embedding_type, text_type=text_type,
                                     embedding=embedding)

        return drug_embedding

    def find_drug_embedding(self, drug_id: int, embedding_type: EmbeddingType, text_type: TextType) \
            -> DrugEmbedding:
        result, _ = self.call_procedure('FindDrugEmbedding',
                                        [drug_id, embedding_type.value, text_type.value])

        drug_embedding = result[0]

        return drug_embedding

    def is_exists_drug_embedding(self, drug_id: int, embedding_type: EmbeddingType, text_type: TextType) \
            -> bool:
        result, _ = self.call_procedure('FindDrugEmbedding',
                                        [drug_id, embedding_type.value, text_type.value])

        similarity = result[0]

        return similarity is not None and (similarity != [] if isinstance(similarity, list) else bool(similarity))

    def find_all_embedding(self, embedding_type: EmbeddingType, text_type: TextType) \
            -> list[DrugEmbeddingDTO]:
        result, _ = self.call_procedure('FindAllEmbedding', [embedding_type.value, text_type.value])

        return map_drug_embedding(result[0])

    def find_all_embedding_dict(self, embedding_type: EmbeddingType, text_type: TextType) \
            -> dict:
        result, _ = self.call_procedure('FindAllEmbedding', [embedding_type.value, text_type.value])

        return map_drug_embedding_dict(result[0])

    def find_drug_embedding_description(self, embedding_type: EmbeddingType, start: int, length: int) \
            -> list[TextEmbeddingDTO]:
        result, _ = self.call_procedure('FindDrugEmbeddingDescription', [embedding_type.value, start, length])

        return map_text_embedding(result[0])

    def find_drug_embedding_indication(self, embedding_type: EmbeddingType, start: int, length: int) \
            -> list[TextEmbeddingDTO]:
        result, _ = self.call_procedure('FindDrugEmbeddingIndication', [embedding_type.value, start, length])

        return map_text_embedding(result[0])

    def find_drug_embedding_pharmacodynamics(self, embedding_type: EmbeddingType, start: int, length: int) \
            -> list[TextEmbeddingDTO]:
        result, _ = self.call_procedure('FindDrugEmbeddingPharmacodynamics', [embedding_type.value, start, length])

        return map_text_embedding(result[0])

    def find_drug_embedding_mechanism_of_action(self, embedding_type: EmbeddingType, start: int, length: int) \
            -> list[TextEmbeddingDTO]:
        result, _ = self.call_procedure('FindDrugEmbeddingMechanismOfAction', [embedding_type.value, start, length])

        return map_text_embedding(result[0])

    def find_drug_embedding_toxicity(self, embedding_type: EmbeddingType, start: int, length: int) \
            -> list[TextEmbeddingDTO]:
        result, _ = self.call_procedure('FindDrugEmbeddingToxicity', [embedding_type.value, start, length])

        return map_text_embedding(result[0])

    def find_drug_embedding_metabolism(self, embedding_type: EmbeddingType, start: int, length: int) \
            -> list[TextEmbeddingDTO]:
        result, _ = self.call_procedure('FindDrugEmbeddingMetabolism', [embedding_type.value, start, length])

        return map_text_embedding(result[0])

    def find_drug_embedding_absorption(self, embedding_type: EmbeddingType, start: int, length: int) \
            -> list[TextEmbeddingDTO]:
        result, _ = self.call_procedure('FindDrugEmbeddingAbsorption', [embedding_type.value, start, length])

        return map_text_embedding(result[0])

    def find_drug_embedding_half_life(self, embedding_type: EmbeddingType, start: int, length: int) \
            -> list[TextEmbeddingDTO]:
        result, _ = self.call_procedure('FindDrugEmbeddingHalfLife', [embedding_type.value, start, length])

        return map_text_embedding(result[0])

    def find_drug_embedding_protein_binding(self, embedding_type: EmbeddingType, start: int, length: int) \
            -> list[TextEmbeddingDTO]:
        result, _ = self.call_procedure('FindDrugEmbeddingProteinBinding', [embedding_type.value, start, length])

        return map_text_embedding(result[0])

    def find_drug_embedding_route_of_elimination(self, embedding_type: EmbeddingType, start: int, length: int) \
            -> list[TextEmbeddingDTO]:
        result, _ = self.call_procedure('FindDrugEmbeddingRouteOfElimination', [embedding_type.value, start, length])

        return map_text_embedding(result[0])

    def find_drug_embedding_volume_of_distribution(self, embedding_type: EmbeddingType, start: int, length: int) \
            -> list[TextEmbeddingDTO]:
        result, _ = self.call_procedure('FindDrugEmbeddingVolumeOfDistribution', [embedding_type.value, start, length])

        return map_text_embedding(result[0])

    def find_drug_embedding_clearance(self, embedding_type: EmbeddingType, start: int, length: int) \
            -> list[TextEmbeddingDTO]:
        result, _ = self.call_procedure('FindDrugEmbeddingClearance', [embedding_type.value, start, length])

        return map_text_embedding(result[0])

    def find_drug_embedding_classification_description(self, embedding_type: EmbeddingType, start: int, length: int) \
            -> list[TextEmbeddingDTO]:
        result, _ = self.call_procedure('FindDrugEmbeddingClassificationDescription',
                                        [embedding_type.value, start, length])

        return map_text_embedding(result[0])

    def get_embedding_count(self, embedding_type: EmbeddingType, text_type: TextType):

        result, _ = self.call_procedure('GetEmbeddingCount', [embedding_type.value, text_type.value])

        return result[0]
