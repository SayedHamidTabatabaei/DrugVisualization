from dataclasses import dataclass, field
import numpy as np


@dataclass
class TrainingDataDTO:
    drug_1: int
    drugbank_id_1: int
    reduction_values_1: list[float]
    drug_2: int
    drugbank_id_2: int
    reduction_values_2: list[float]
    interaction_type: int
    concat_values: np.ndarray = field(init=False)

    # This method runs after the dataclass is initialized
    def __post_init__(self):
        # Perform the concatenation here
        self.concat_values = np.concatenate((self.reduction_values_1, self.reduction_values_2))
