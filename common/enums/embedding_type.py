from enum import Enum


class EmbeddingType(Enum):
    PubMedBERT = 1
    SciBERT = 2
    PubMedBERT_32 = 33
    SciBERT_32 = 34
    PubMedBERT_64 = 65
    SciBERT_64 = 66
    PubMedBERT_128 = 129
    SciBERT_128 = 130
    PubMedBERT_256 = 257
    SciBERT_256 = 258
    PubMedBERT_512 = 513
    SciBERT_512 = 514
