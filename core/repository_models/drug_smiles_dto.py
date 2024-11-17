from dataclasses import dataclass

from typing import Union
from numpy.typing import NDArray


@dataclass
class DrugSmilesDTO:
    id: int
    drugbank_id: str
    smiles: str
    has_enzyme: bool
    has_pathway: bool
    has_target: bool
    has_smiles: bool
    fingerprint: Union[str, list[NDArray]]
