from dataclasses import dataclass


@dataclass
class DrugMetabolismDTO:
    id: int
    metabolism: str
