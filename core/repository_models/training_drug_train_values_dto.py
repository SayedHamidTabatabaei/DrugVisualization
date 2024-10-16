from dataclasses import dataclass

from common.enums.category import Category


@dataclass
class TrainingDrugTrainValuesDTO:
    category: Category
    values: list[float] | dict[int, float]
