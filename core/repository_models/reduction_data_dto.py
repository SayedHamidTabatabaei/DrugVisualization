from dataclasses import dataclass

from common.enums.category import Category
from common.enums.reduction_category import ReductionCategory
from common.enums.similarity_type import SimilarityType


@dataclass
class ReductionDataDTO:
    id: int
    drug_id: int
    drugbank_id: str
    similarity_type: SimilarityType
    category: Category
    reduction_category: ReductionCategory
    reduction_value: str
    has_enzyme: bool
    has_pathway: bool
    has_target: bool
    has_smiles: bool
