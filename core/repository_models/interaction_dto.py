from dataclasses import dataclass


@dataclass
class InteractionDTO:
    drug_1: int
    drugbank_id_1: int
    drug_2: int
    drugbank_id_2: int
    interaction_type: int
