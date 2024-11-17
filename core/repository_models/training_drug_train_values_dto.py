from dataclasses import dataclass

from typing import Union, Dict
from numpy.typing import NDArray

from common.enums.category import Category


@dataclass
class TrainingDrugTrainValuesDTO:
    category: Category
    values: Union[NDArray, Dict[int, float]]
