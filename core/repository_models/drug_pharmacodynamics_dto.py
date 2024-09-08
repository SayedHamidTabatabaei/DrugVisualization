from dataclasses import dataclass


@dataclass
class DrugPharmacodynamicsDTO:
    id: int
    pharmacodynamics: str
