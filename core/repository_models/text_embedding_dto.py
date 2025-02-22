from dataclasses import dataclass


@dataclass
class TextEmbeddingDTO:
    id: int
    drug_id: int
    drugbank_id: str
    drug_name: str
    embedding_type: int
    text_type: int
    embedding: str
    text: str
