from common.enums.category import Category
from common.enums.embedding_type import EmbeddingType
from common.enums.text_type import TextType


def find_category(embedding_type: EmbeddingType, text_type: TextType) -> Category:
    if embedding_type == EmbeddingType.PubMedBERT:
        if text_type == TextType.Description:
            return Category.PubmedBertDescription
        elif text_type == TextType.Indication:
            return Category.PubmedBertIndication
        elif text_type == TextType.Pharmacodynamics:
            return Category.PubmedBertPharmacodynamics
        elif text_type == TextType.MechanismOfAction:
            return Category.PubmedBertMechanismOfAction
        elif text_type == TextType.Toxicity:
            return Category.PubmedBertToxicity
        elif text_type == TextType.Metabolism:
            return Category.PubmedBertMetabolism
        elif text_type == TextType.Absorption:
            return Category.PubmedBertAbsorption
        elif text_type == TextType.HalfLife:
            return Category.PubmedBertHalfLife
        elif text_type == TextType.ProteinBinding:
            return Category.PubmedBertProteinBinding
        elif text_type == TextType.RouteOfElimination:
            return Category.PubmedBertRouteOfElimination
        elif text_type == TextType.VolumeOfDistribution:
            return Category.PubmedBertVolumeOfDistribution
        elif text_type == TextType.Clearance:
            return Category.PubmedBertClearance
        elif text_type == TextType.ClassificationDescription:
            return Category.PubmedBertClassificationDescription
        elif text_type == TextType.TotalText:
            return Category.PubmedBertTotalText

    elif embedding_type == EmbeddingType.SciBERT:
        if text_type == TextType.Description:
            return Category.SciBertDescription
        elif text_type == TextType.Indication:
            return Category.SciBertIndication
        elif text_type == TextType.Pharmacodynamics:
            return Category.SciBertPharmacodynamics
        elif text_type == TextType.MechanismOfAction:
            return Category.SciBertMechanismOfAction
        elif text_type == TextType.Toxicity:
            return Category.SciBertToxicity
        elif text_type == TextType.Metabolism:
            return Category.SciBertMetabolism
        elif text_type == TextType.Absorption:
            return Category.SciBertAbsorption
        elif text_type == TextType.HalfLife:
            return Category.SciBertHalfLife
        elif text_type == TextType.ProteinBinding:
            return Category.SciBertProteinBinding
        elif text_type == TextType.RouteOfElimination:
            return Category.SciBertRouteOfElimination
        elif text_type == TextType.VolumeOfDistribution:
            return Category.SciBertVolumeOfDistribution
        elif text_type == TextType.Clearance:
            return Category.SciBertClearance
        elif text_type == TextType.ClassificationDescription:
            return Category.SciBertClassificationDescription
        elif text_type == TextType.TotalText:
            return Category.SciBertTotalText

    elif embedding_type == EmbeddingType.LongFormer_BioNER:
        if text_type == TextType.Description:
            return Category.LongFormerDescription
        elif text_type == TextType.Indication:
            return Category.LongFormerIndication
        elif text_type == TextType.Pharmacodynamics:
            return Category.LongFormerPharmacodynamics
        elif text_type == TextType.MechanismOfAction:
            return Category.LongFormerMechanismOfAction
        elif text_type == TextType.Toxicity:
            return Category.LongFormerToxicity
        elif text_type == TextType.Metabolism:
            return Category.LongFormerMetabolism
        elif text_type == TextType.Absorption:
            return Category.LongFormerAbsorption
        elif text_type == TextType.HalfLife:
            return Category.LongFormerHalfLife
        elif text_type == TextType.ProteinBinding:
            return Category.LongFormerProteinBinding
        elif text_type == TextType.RouteOfElimination:
            return Category.LongFormerRouteOfElimination
        elif text_type == TextType.VolumeOfDistribution:
            return Category.LongFormerVolumeOfDistribution
        elif text_type == TextType.Clearance:
            return Category.LongFormerClearance
        elif text_type == TextType.ClassificationDescription:
            return Category.LongFormerClassificationDescription
        elif text_type == TextType.TotalText:
            return Category.LongFormerTotalText

    elif embedding_type == EmbeddingType.BigBird_PubMed:
        if text_type == TextType.Description:
            return Category.BigBirdDescription
        elif text_type == TextType.Indication:
            return Category.BigBirdIndication
        elif text_type == TextType.Pharmacodynamics:
            return Category.BigBirdPharmacodynamics
        elif text_type == TextType.MechanismOfAction:
            return Category.BigBirdMechanismOfAction
        elif text_type == TextType.Toxicity:
            return Category.BigBirdToxicity
        elif text_type == TextType.Metabolism:
            return Category.BigBirdMetabolism
        elif text_type == TextType.Absorption:
            return Category.BigBirdAbsorption
        elif text_type == TextType.HalfLife:
            return Category.BigBirdHalfLife
        elif text_type == TextType.ProteinBinding:
            return Category.BigBirdProteinBinding
        elif text_type == TextType.RouteOfElimination:
            return Category.BigBirdRouteOfElimination
        elif text_type == TextType.VolumeOfDistribution:
            return Category.BigBirdVolumeOfDistribution
        elif text_type == TextType.Clearance:
            return Category.BigBirdClearance
        elif text_type == TextType.ClassificationDescription:
            return Category.BigBirdClassificationDescription
        elif text_type == TextType.TotalText:
            return Category.BigBirdTotalText

    elif embedding_type == EmbeddingType.LED:
        if text_type == TextType.Description:
            return Category.LEDDescription
        elif text_type == TextType.Indication:
            return Category.LEDIndication
        elif text_type == TextType.Pharmacodynamics:
            return Category.LEDPharmacodynamics
        elif text_type == TextType.MechanismOfAction:
            return Category.LEDMechanismOfAction
        elif text_type == TextType.Toxicity:
            return Category.LEDToxicity
        elif text_type == TextType.Metabolism:
            return Category.LEDMetabolism
        elif text_type == TextType.Absorption:
            return Category.LEDAbsorption
        elif text_type == TextType.HalfLife:
            return Category.LEDHalfLife
        elif text_type == TextType.ProteinBinding:
            return Category.LEDProteinBinding
        elif text_type == TextType.RouteOfElimination:
            return Category.LEDRouteOfElimination
        elif text_type == TextType.VolumeOfDistribution:
            return Category.LEDVolumeOfDistribution
        elif text_type == TextType.Clearance:
            return Category.LEDClearance
        elif text_type == TextType.ClassificationDescription:
            return Category.LEDClassificationDescription
        elif text_type == TextType.TotalText:
            return Category.LEDTotalText

    raise ValueError(f"Unsupported combination of embedding_type: {embedding_type} and text_type: {text_type}")
