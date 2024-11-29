from dataclasses import dataclass

from typing import Union, Dict
from numpy.typing import NDArray

from common.enums.category import Category
from common.enums.similarity_type import SimilarityType


@dataclass
class TrainingDrugTrainValuesDTO:
    category: Category
    similarity_type: SimilarityType
    values: Union[NDArray, Dict[int, float]]
