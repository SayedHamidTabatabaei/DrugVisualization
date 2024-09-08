from dataclasses import dataclass


@dataclass
class DrugClearanceDTO:
    id: int
    clearance: str
