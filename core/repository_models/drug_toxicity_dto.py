from dataclasses import dataclass


@dataclass
class DrugToxicityDTO:
    id: int
    toxicity: str
