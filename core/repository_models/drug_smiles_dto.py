from dataclasses import dataclass


@dataclass
class DrugSmilesDTO:
    id: int
    smiles: str
    has_enzyme: bool
    has_pathway: bool
    has_target: bool
    has_smiles: bool
    fingerprint: str
