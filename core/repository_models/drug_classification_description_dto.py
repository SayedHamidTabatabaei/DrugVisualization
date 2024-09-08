from dataclasses import dataclass


@dataclass
class DrugClassificationDescriptionDTO:
    id: int
    classification_description: str
