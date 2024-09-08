from dataclasses import dataclass


@dataclass
class DrugMechanismOfActionDTO:
    id: int
    mechanism_of_action: str
