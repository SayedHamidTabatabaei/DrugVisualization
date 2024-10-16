from dataclasses import dataclass


@dataclass
class InteractionEmbeddingDTO:
    id: int
    drug1_id: int
    drugbank1_id: str
    drug2_id: int
    drugbank2_id: str
    embedding_type: int
    text_type: int
    embedding: str
