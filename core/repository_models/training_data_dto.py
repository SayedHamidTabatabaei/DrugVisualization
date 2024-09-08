from dataclasses import dataclass


@dataclass
class TrainingDataDTO:
    drug_1: int
    drugbank_id_1: int
    reduction_values_1: list[float]
    drug_2: int
    drugbank_id_2: int
    reduction_values_2: list[float]
    interaction_type: int
