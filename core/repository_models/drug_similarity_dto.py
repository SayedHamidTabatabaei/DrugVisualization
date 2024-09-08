from dataclasses import dataclass

from common.enums.category import Category
from common.enums.similarity_type import SimilarityType


@dataclass
class DrugSimilarityDTO:
    similarity_type: SimilarityType
    category: Category
    drug_1: int
    drug_2: int
    drugbank_id_1: str
    drugbank_id_2: str
    value: float
