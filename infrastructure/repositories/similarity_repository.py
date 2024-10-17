from decimal import Decimal
from typing import Union
import json

from common.enums.category import Category
from common.enums.similarity_type import SimilarityType
from core.domain.similarity import Similarity
from core.mappers.drug_similarity_mapper import map_drug_similarity
from core.repository_models.drug_similarity_dto import DrugSimilarityDTO
from infrastructure.mysqldb.mysql_repository import MySqlRepository


class SimilarityRepository(MySqlRepository):
    def __init__(self):
        super().__init__('similarities')

    def insert(self, similarity_type: SimilarityType, category: Category, drug_1: int, drug_2: int, value: Decimal) \
            -> Similarity:
        similarity = Similarity(similarity_type=similarity_type, category=category, drug_1=drug_1, drug_2=drug_2,
                                value=value)

        super().insert(similarity)

        return similarity

    def insert_if_not_exits(self, similarity_type: SimilarityType, category: Category, drug_1: int, drug_2: int,
                            value: Decimal) -> Union[Similarity, None]:
        is_exists = self.is_exists_similarity(similarity_type, category, drug_1, drug_2)

        if is_exists:
            return None

        similarity = self.insert(similarity_type=similarity_type, category=category, drug_1=drug_1, drug_2=drug_2,
                                 value=value)

        return similarity

    def insert_batch_check_duplicate(self, similarities: list[Similarity]):
        super().insert_batch_check_duplicate(similarities, [Similarity.value])

    def find_similarity(self, similarity_type: SimilarityType, category: Category, drug_1: int, drug_2: int) \
            -> Similarity:
        similarity, _ = self.call_procedure('FindSimilarity',
                                            [similarity_type.value, category.value, drug_1, drug_2])

        return similarity

    def is_exists_similarity(self, similarity_type: SimilarityType, category: Category, drug_1: int, drug_2: int) \
            -> bool:
        result, _ = self.call_procedure('FindSimilarity',
                                        [similarity_type.value, category.value, drug_1, drug_2])

        similarity = result[0]

        return similarity is not None and (similarity != [] if isinstance(similarity, list) else bool(similarity))

    def find_all_similarity(self, similarity_type: SimilarityType, category: Category,
                            check_target: bool = True, check_pathway: bool = True, check_enzyme: bool = True, check_smiles: bool = True,
                            start: int = 0, length: int = 100000) \
            -> list[DrugSimilarityDTO]:
        result, _ = self.call_procedure('FindAllSimilarities',
                                        [start, length, similarity_type.value, category.value,
                                         check_target, check_pathway, check_enzyme, check_smiles])

        similarity = result[0]

        return map_drug_similarity(similarity)

    def find_all_active_similarity(self, similarity_type: SimilarityType, category: Category) \
            -> list[DrugSimilarityDTO]:
        result, _ = self.call_procedure('FindAllActiveSimilarities', [similarity_type.value, category.value])

        similarity = result[0]

        return map_drug_similarity(similarity)

    def find_similarities_by_drugs(self, drug_ids: list[int], similarity_type: SimilarityType, category: Category) -> list[DrugSimilarityDTO]:
        ids_json = json.dumps(drug_ids)

        result, _ = self.call_procedure('FindSimilaritiesByDrugs', [ids_json, similarity_type.value, category.value])

        similarity = result[0]

        return map_drug_similarity(similarity)

    def get_similarity_count(self, similarity_type: SimilarityType, category: Category,
                             check_target: bool, check_pathway: bool, check_enzyme: bool, check_smiles: bool) -> int:

        result, _ = self.call_procedure('GetSimilarityCount',
                                        [similarity_type.value, category.value,
                                         check_target, check_pathway, check_enzyme, check_smiles])

        return result[0]

    def find_exists_similarity_types_by_category(self, category: Category):

        result, _ = self.call_procedure('FindExistsSimilarityTypesByCategory', [category.value])

        return [SimilarityType(r[0]) for r in result[0]]
