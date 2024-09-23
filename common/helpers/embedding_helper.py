from common.enums.category import Category
from common.enums.embedding_type import EmbeddingType
from common.enums.text_type import TextType


def find_category(embedding_type: EmbeddingType, text_type: TextType) -> Category:
    match (embedding_type, text_type):
        case (EmbeddingType.PubMedBERT, TextType.Description):
            return Category.PubmedBertDescription
        case (EmbeddingType.PubMedBERT, TextType.Indication):
            return Category.PubmedBertIndication
        case (EmbeddingType.PubMedBERT, TextType.Pharmacodynamics):
            return Category.PubmedBertPharmacodynamics
        case (EmbeddingType.PubMedBERT, TextType.MechanismOfAction):
            return Category.PubmedBertMechanismOfAction
        case (EmbeddingType.PubMedBERT, TextType.Toxicity):
            return Category.PubmedBertToxicity
        case (EmbeddingType.PubMedBERT, TextType.Metabolism):
            return Category.PubmedBertMetabolism
        case (EmbeddingType.PubMedBERT, TextType.Absorption):
            return Category.PubmedBertAbsorption
        case (EmbeddingType.PubMedBERT, TextType.HalfLife):
            return Category.PubmedBertHalfLife
        case (EmbeddingType.PubMedBERT, TextType.ProteinBinding):
            return Category.PubmedBertProteinBinding
        case (EmbeddingType.PubMedBERT, TextType.RouteOfElimination):
            return Category.PubmedBertRouteOfElimination
        case (EmbeddingType.PubMedBERT, TextType.VolumeOfDistribution):
            return Category.PubmedBertVolumeOfDistribution
        case (EmbeddingType.PubMedBERT, TextType.Clearance):
            return Category.PubmedBertClearance
        case (EmbeddingType.PubMedBERT, TextType.ClassificationDescription):
            return Category.PubmedBertClassificationDescription
        case (EmbeddingType.SciBERT, TextType.Description):
            return Category.SciBertDescription
        case (EmbeddingType.SciBERT, TextType.Indication):
            return Category.SciBertIndication
        case (EmbeddingType.SciBERT, TextType.Pharmacodynamics):
            return Category.SciBertPharmacodynamics
        case (EmbeddingType.SciBERT, TextType.MechanismOfAction):
            return Category.SciBertMechanismOfAction
        case (EmbeddingType.SciBERT, TextType.Toxicity):
            return Category.SciBertToxicity
        case (EmbeddingType.SciBERT, TextType.Metabolism):
            return Category.SciBertMetabolism
        case (EmbeddingType.SciBERT, TextType.Absorption):
            return Category.SciBertAbsorption
        case (EmbeddingType.SciBERT, TextType.HalfLife):
            return Category.SciBertHalfLife
        case (EmbeddingType.SciBERT, TextType.ProteinBinding):
            return Category.SciBertProteinBinding
        case (EmbeddingType.SciBERT, TextType.RouteOfElimination):
            return Category.SciBertRouteOfElimination
        case (EmbeddingType.SciBERT, TextType.VolumeOfDistribution):
            return Category.SciBertVolumeOfDistribution
        case (EmbeddingType.SciBERT, TextType.Clearance):
            return Category.SciBertClearance
        case (EmbeddingType.SciBERT, TextType.ClassificationDescription):
            return Category.SciBertClassificationDescription
        case _:
            raise
