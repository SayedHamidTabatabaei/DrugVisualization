from dataclasses import dataclass


@dataclass
class TrainingDrugInteractionDTO:
    drug_1: int
    drug_2: int
    interaction_type: int
