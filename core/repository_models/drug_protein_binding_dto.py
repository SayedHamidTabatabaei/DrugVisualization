from dataclasses import dataclass


@dataclass
class DrugProteinBindingDTO:
    id: int
    protein_binding: str
