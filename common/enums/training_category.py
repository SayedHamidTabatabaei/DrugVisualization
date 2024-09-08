from enum import Enum


class TrainingCategory(Enum):
    Substructure = 1,
    Target = 2,
    Pathway = 3,
    Enzyme = 4,
    PubmedBERTDescription = 11
    SciBERTDescription = 31
    Total = 100
