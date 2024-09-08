from dataclasses import dataclass


@dataclass
class DrugDescriptionDTO:
    id: int
    description: str
