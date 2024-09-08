from dataclasses import dataclass


@dataclass
class DrugSmilesDTO:
    id: int
    smiles: str
    fingerprint: str
