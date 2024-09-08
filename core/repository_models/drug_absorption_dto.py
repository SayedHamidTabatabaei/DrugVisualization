from dataclasses import dataclass


@dataclass
class DrugAbsorptionDTO:
    id: int
    absorption: str
