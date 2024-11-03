from enum import Enum


class Category(Enum):
    Substructure = (1, object)
    Target = (2, object)
    Pathway = (3, object)
    Enzyme = (4, object)
    PubmedBertDescription = (11, str)
    PubmedBertIndication = (12, str)
    PubmedBertPharmacodynamics = (13, str)
    PubmedBertMechanismOfAction = (14, str)
    PubmedBertToxicity = (15, str)
    PubmedBertMetabolism = (16, str)
    PubmedBertAbsorption = (17, str)
    PubmedBertHalfLife = (18, str)
    PubmedBertProteinBinding = (19, str)
    PubmedBertRouteOfElimination = (20, str)
    PubmedBertVolumeOfDistribution = (21, str)
    PubmedBertClearance = (22, str)
    PubmedBertClassificationDescription = (23, str)
    SciBertDescription = (31, str)
    SciBertIndication = (32, str)
    SciBertPharmacodynamics = (33, str)
    SciBertMechanismOfAction = (34, str)
    SciBertToxicity = (35, str)
    SciBertMetabolism = (36, str)
    SciBertAbsorption = (37, str)
    SciBertHalfLife = (38, str)
    SciBertProteinBinding = (39, str)
    SciBertRouteOfElimination = (40, str)
    SciBertVolumeOfDistribution = (41, str)
    SciBertClearance = (42, str)
    SciBertClassificationDescription = (43, str)

    def __init__(self, value, data_type: type):
        self._value_ = value
        self.data_type = data_type

    @classmethod
    def from_value(cls, value):
        return next((c for c in cls if c.value == value), None)
