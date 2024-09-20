from dataclasses import dataclass

from common.enums.category import Category
from common.enums.reduction_category import ReductionCategory
from common.enums.similarity_type import SimilarityType


@dataclass
class ReductionParameterModel:
    similarity_type: SimilarityType
    category: Category
    reduction_category: ReductionCategory
