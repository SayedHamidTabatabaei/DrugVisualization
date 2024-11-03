from dataclasses import dataclass


@dataclass
class TrainingDrugInteractionDTO:
    id: int
    drug_1: int
    drug_2: int
    interaction_type: int
    interaction_description: list
