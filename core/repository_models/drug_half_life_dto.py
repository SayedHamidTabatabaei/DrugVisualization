from dataclasses import dataclass


@dataclass
class DrugHalfLifeDTO:
    id: int
    half_life: str
