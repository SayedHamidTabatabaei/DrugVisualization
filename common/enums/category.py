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
    PubmedBertTotalText = (24, str)
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
    SciBertTotalText = (44, str)
    LongFormerDescription = (51, str)
    LongFormerIndication = (52, str)
    LongFormerPharmacodynamics = (53, str)
    LongFormerMechanismOfAction = (54, str)
    LongFormerToxicity = (55, str)
    LongFormerMetabolism = (56, str)
    LongFormerAbsorption = (57, str)
    LongFormerHalfLife = (58, str)
    LongFormerProteinBinding = (59, str)
    LongFormerRouteOfElimination = (60, str)
    LongFormerVolumeOfDistribution = (61, str)
    LongFormerClearance = (62, str)
    LongFormerClassificationDescription = (63, str)
    LongFormerTotalText = (64, str)
    BigBirdDescription = (71, str)
    BigBirdIndication = (72, str)
    BigBirdPharmacodynamics = (73, str)
    BigBirdMechanismOfAction = (74, str)
    BigBirdToxicity = (75, str)
    BigBirdMetabolism = (76, str)
    BigBirdAbsorption = (77, str)
    BigBirdHalfLife = (78, str)
    BigBirdProteinBinding = (79, str)
    BigBirdRouteOfElimination = (80, str)
    BigBirdVolumeOfDistribution = (81, str)
    BigBirdClearance = (82, str)
    BigBirdClassificationDescription = (83, str)
    BigBirdTotalText = (84, str)
    LEDDescription = (91, str)
    LEDIndication = (92, str)
    LEDPharmacodynamics = (93, str)
    LEDMechanismOfAction = (94, str)
    LEDToxicity = (95, str)
    LEDMetabolism = (96, str)
    LEDAbsorption = (97, str)
    LEDHalfLife = (98, str)
    LEDProteinBinding = (99, str)
    LEDRouteOfElimination = (100, str)
    LEDVolumeOfDistribution = (101, str)
    LEDClearance = (102, str)
    LEDClassificationDescription = (103, str)
    LEDTotalText = (104, str)

    def __init__(self, value, data_type: type):
        self._value_ = value
        self.data_type = data_type

    @classmethod
    def from_value(cls, value):
        return next((c for c in cls if c.value == value), None)
