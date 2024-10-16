from dataclasses import dataclass, field
import numpy as np

from common.enums.category import Category


@dataclass
class TrainingInteractionDataDTO:
    drug_1: int
    drugbank_id_1: int
    reduction_values_1: list
    drug_2: int
    drugbank_id_2: int
    reduction_values_2: list
    category: Category
    interaction_type: int
    concat_values: np.ndarray = field(init=False)

    # This method runs after the dataclass is initialized
    def __post_init__(self):
        self.concat_values = np.concatenate((self.reduction_values_1, self.reduction_values_2))
