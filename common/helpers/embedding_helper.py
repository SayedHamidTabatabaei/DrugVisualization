from common.enums.category import Category
from common.enums.embedding_type import EmbeddingType
from common.enums.text_type import TextType


def find_category(embedding_type: EmbeddingType, text_type: TextType) -> Category:
    pubmed_bert_types = {
        EmbeddingType.PubMedBERT, EmbeddingType.PubMedBERT_32, EmbeddingType.PubMedBERT_64,
        EmbeddingType.PubMedBERT_128, EmbeddingType.PubMedBERT_256, EmbeddingType.PubMedBERT_512
    }
    scibert_types = {
        EmbeddingType.SciBERT, EmbeddingType.SciBERT_32, EmbeddingType.SciBERT_64,
        EmbeddingType.SciBERT_128, EmbeddingType.SciBERT_256, EmbeddingType.SciBERT_512
    }

    if embedding_type in pubmed_bert_types:
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

    elif embedding_type in scibert_types:
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

    raise ValueError(f"Unsupported combination of embedding_type: {embedding_type} and text_type: {text_type}")
