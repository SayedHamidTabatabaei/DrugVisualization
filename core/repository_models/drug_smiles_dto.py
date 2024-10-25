from dataclasses import dataclass

from numpy import ndarray


@dataclass
class DrugSmilesDTO:
    id: int
    drugbank_id: str
    smiles: str
    has_enzyme: bool
    has_pathway: bool
    has_target: bool
    has_smiles: bool
    fingerprint: str | list[ndarray]
