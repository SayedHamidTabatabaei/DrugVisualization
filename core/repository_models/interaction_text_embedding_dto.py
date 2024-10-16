from dataclasses import dataclass


@dataclass
class InteractionTextEmbeddingDTO:
    id: int
    drug1_id: int
    drugbank1_id: str
    drug1_name: str
    drug2_id: int
    drugbank2_id: str
    drug2_name: str
    embedding_type: int
    text_type: int
    embedding: str
    text: str
