from dataclasses import dataclass


@dataclass
class DrugIndicationDTO:
    id: int
    indication: str
