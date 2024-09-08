from dataclasses import dataclass


@dataclass
class DrugEmbeddingDTO:
    id: int
    drug_id: int
    drugbank_id: str
    embedding_type: int
    text_type: int
    embedding: str
