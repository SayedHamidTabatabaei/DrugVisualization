from enum import Enum


class Category(Enum):
    Substructure = 1
    Target = 2
    Pathway = 3
    Enzyme = 4
    PubmedBertDescription = 11
    PubmedBertIndication = 12
    PubmedBertPharmacodynamics = 13
    PubmedBertMechanismOfAction = 14
    PubmedBertToxicity = 15
    PubmedBertMetabolism = 16
    PubmedBertAbsorption = 17
    PubmedBertHalfLife = 18
    PubmedBertProteinBinding = 19
    PubmedBertRouteOfElimination = 20
    PubmedBertVolumeOfDistribution = 21
    PubmedBertClearance = 22
    PubmedBertClassificationDescription = 23
    SciBertDescription = 31
    SciBertIndication = 32
    SciBertPharmacodynamics = 33
    SciBertMechanismOfAction = 34
    SciBertToxicity = 35
    SciBertMetabolism = 36
    SciBertAbsorption = 37
    SciBertHalfLife = 38
    SciBertProteinBinding = 39
    SciBertRouteOfElimination = 40
    SciBertVolumeOfDistribution = 41
    SciBertClearance = 42
    SciBertClassificationDescription = 43
