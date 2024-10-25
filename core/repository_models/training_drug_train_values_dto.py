from dataclasses import dataclass

from numpy import ndarray

from common.enums.category import Category


@dataclass
class TrainingDrugTrainValuesDTO:
    category: Category
    values: ndarray | dict[int, float]
