from enum import Enum


class EmbeddingType(Enum):
    PubMedBERT = 1
    SciBERT = 2
    LongFormer_BioNER = 3
    BigBird_PubMed = 4
    LED = 5
