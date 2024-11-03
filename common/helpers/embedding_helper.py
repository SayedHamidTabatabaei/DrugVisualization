from common.enums.category import Category
from common.enums.embedding_type import EmbeddingType
from common.enums.text_type import TextType


def find_category(embedding_type: EmbeddingType, text_type: TextType) -> Category:
    match (embedding_type, text_type):
        case ((EmbeddingType.PubMedBERT | EmbeddingType.PubMedBERT_32 | EmbeddingType.PubMedBERT_64 | EmbeddingType.PubMedBERT_128 |
               EmbeddingType.PubMedBERT_256 | EmbeddingType.PubMedBERT_512), TextType.Description):
            return Category.PubmedBertDescription
        case ((EmbeddingType.PubMedBERT | EmbeddingType.PubMedBERT_32 | EmbeddingType.PubMedBERT_64 | EmbeddingType.PubMedBERT_128 |
               EmbeddingType.PubMedBERT_256 | EmbeddingType.PubMedBERT_512), TextType.Indication):
            return Category.PubmedBertIndication
        case ((EmbeddingType.PubMedBERT | EmbeddingType.PubMedBERT_32 | EmbeddingType.PubMedBERT_64 | EmbeddingType.PubMedBERT_128 |
               EmbeddingType.PubMedBERT_256 | EmbeddingType.PubMedBERT_512), TextType.Pharmacodynamics):
            return Category.PubmedBertPharmacodynamics
        case ((EmbeddingType.PubMedBERT | EmbeddingType.PubMedBERT_32 | EmbeddingType.PubMedBERT_64 | EmbeddingType.PubMedBERT_128 |
               EmbeddingType.PubMedBERT_256 | EmbeddingType.PubMedBERT_512), TextType.MechanismOfAction):
            return Category.PubmedBertMechanismOfAction
        case ((EmbeddingType.PubMedBERT | EmbeddingType.PubMedBERT_32 | EmbeddingType.PubMedBERT_64 | EmbeddingType.PubMedBERT_128 |
               EmbeddingType.PubMedBERT_256 | EmbeddingType.PubMedBERT_512), TextType.Toxicity):
            return Category.PubmedBertToxicity
        case ((EmbeddingType.PubMedBERT | EmbeddingType.PubMedBERT_32 | EmbeddingType.PubMedBERT_64 | EmbeddingType.PubMedBERT_128 |
               EmbeddingType.PubMedBERT_256 | EmbeddingType.PubMedBERT_512), TextType.Metabolism):
            return Category.PubmedBertMetabolism
        case ((EmbeddingType.PubMedBERT | EmbeddingType.PubMedBERT_32 | EmbeddingType.PubMedBERT_64 | EmbeddingType.PubMedBERT_128 |
               EmbeddingType.PubMedBERT_256 | EmbeddingType.PubMedBERT_512), TextType.Absorption):
            return Category.PubmedBertAbsorption
        case ((EmbeddingType.PubMedBERT | EmbeddingType.PubMedBERT_32 | EmbeddingType.PubMedBERT_64 | EmbeddingType.PubMedBERT_128 |
               EmbeddingType.PubMedBERT_256 | EmbeddingType.PubMedBERT_512), TextType.HalfLife):
            return Category.PubmedBertHalfLife
        case ((EmbeddingType.PubMedBERT | EmbeddingType.PubMedBERT_32 | EmbeddingType.PubMedBERT_64 | EmbeddingType.PubMedBERT_128 |
               EmbeddingType.PubMedBERT_256 | EmbeddingType.PubMedBERT_512), TextType.ProteinBinding):
            return Category.PubmedBertProteinBinding
        case ((EmbeddingType.PubMedBERT | EmbeddingType.PubMedBERT_32 | EmbeddingType.PubMedBERT_64 | EmbeddingType.PubMedBERT_128 |
               EmbeddingType.PubMedBERT_256 | EmbeddingType.PubMedBERT_512), TextType.RouteOfElimination):
            return Category.PubmedBertRouteOfElimination
        case ((EmbeddingType.PubMedBERT | EmbeddingType.PubMedBERT_32 | EmbeddingType.PubMedBERT_64 | EmbeddingType.PubMedBERT_128 |
               EmbeddingType.PubMedBERT_256 | EmbeddingType.PubMedBERT_512), TextType.VolumeOfDistribution):
            return Category.PubmedBertVolumeOfDistribution
        case ((EmbeddingType.PubMedBERT | EmbeddingType.PubMedBERT_32 | EmbeddingType.PubMedBERT_64 | EmbeddingType.PubMedBERT_128 |
               EmbeddingType.PubMedBERT_256 | EmbeddingType.PubMedBERT_512), TextType.Clearance):
            return Category.PubmedBertClearance
        case ((EmbeddingType.PubMedBERT | EmbeddingType.PubMedBERT_32 | EmbeddingType.PubMedBERT_64 | EmbeddingType.PubMedBERT_128 |
               EmbeddingType.PubMedBERT_256 | EmbeddingType.PubMedBERT_512), TextType.ClassificationDescription):
            return Category.PubmedBertClassificationDescription
        case ((EmbeddingType.SciBERT | EmbeddingType.SciBERT_32 | EmbeddingType.SciBERT_64 | EmbeddingType.SciBERT_128 | EmbeddingType.SciBERT_256 |
               EmbeddingType.SciBERT_512), TextType.Description):
            return Category.SciBertDescription
        case ((EmbeddingType.SciBERT | EmbeddingType.SciBERT_32 | EmbeddingType.SciBERT_64 | EmbeddingType.SciBERT_128 | EmbeddingType.SciBERT_256 |
               EmbeddingType.SciBERT_512), TextType.Indication):
            return Category.SciBertIndication
        case ((EmbeddingType.SciBERT | EmbeddingType.SciBERT_32 | EmbeddingType.SciBERT_64 | EmbeddingType.SciBERT_128 | EmbeddingType.SciBERT_256 |
               EmbeddingType.SciBERT_512), TextType.Pharmacodynamics):
            return Category.SciBertPharmacodynamics
        case ((EmbeddingType.SciBERT | EmbeddingType.SciBERT_32 | EmbeddingType.SciBERT_64 | EmbeddingType.SciBERT_128 | EmbeddingType.SciBERT_256 |
               EmbeddingType.SciBERT_512), TextType.MechanismOfAction):
            return Category.SciBertMechanismOfAction
        case ((EmbeddingType.SciBERT | EmbeddingType.SciBERT_32 | EmbeddingType.SciBERT_64 | EmbeddingType.SciBERT_128 | EmbeddingType.SciBERT_256 |
               EmbeddingType.SciBERT_512), TextType.Toxicity):
            return Category.SciBertToxicity
        case ((EmbeddingType.SciBERT | EmbeddingType.SciBERT_32 | EmbeddingType.SciBERT_64 | EmbeddingType.SciBERT_128 | EmbeddingType.SciBERT_256 |
               EmbeddingType.SciBERT_512), TextType.Metabolism):
            return Category.SciBertMetabolism
        case ((EmbeddingType.SciBERT | EmbeddingType.SciBERT_32 | EmbeddingType.SciBERT_64 | EmbeddingType.SciBERT_128 | EmbeddingType.SciBERT_256 |
               EmbeddingType.SciBERT_512), TextType.Absorption):
            return Category.SciBertAbsorption
        case ((EmbeddingType.SciBERT | EmbeddingType.SciBERT_32 | EmbeddingType.SciBERT_64 | EmbeddingType.SciBERT_128 | EmbeddingType.SciBERT_256 |
               EmbeddingType.SciBERT_512), TextType.HalfLife):
            return Category.SciBertHalfLife
        case ((EmbeddingType.SciBERT | EmbeddingType.SciBERT_32 | EmbeddingType.SciBERT_64 | EmbeddingType.SciBERT_128 | EmbeddingType.SciBERT_256 |
               EmbeddingType.SciBERT_512), TextType.ProteinBinding):
            return Category.SciBertProteinBinding
        case ((EmbeddingType.SciBERT | EmbeddingType.SciBERT_32 | EmbeddingType.SciBERT_64 | EmbeddingType.SciBERT_128 | EmbeddingType.SciBERT_256 |
               EmbeddingType.SciBERT_512), TextType.RouteOfElimination):
            return Category.SciBertRouteOfElimination
        case ((EmbeddingType.SciBERT | EmbeddingType.SciBERT_32 | EmbeddingType.SciBERT_64 | EmbeddingType.SciBERT_128 | EmbeddingType.SciBERT_256 |
               EmbeddingType.SciBERT_512), TextType.VolumeOfDistribution):
            return Category.SciBertVolumeOfDistribution
        case ((EmbeddingType.SciBERT | EmbeddingType.SciBERT_32 | EmbeddingType.SciBERT_64 | EmbeddingType.SciBERT_128 | EmbeddingType.SciBERT_256 |
               EmbeddingType.SciBERT_512), TextType.Clearance):
            return Category.SciBertClearance
        case ((EmbeddingType.SciBERT | EmbeddingType.SciBERT_32 | EmbeddingType.SciBERT_64 | EmbeddingType.SciBERT_128 | EmbeddingType.SciBERT_256 |
               EmbeddingType.SciBERT_512), TextType.ClassificationDescription):
            return Category.SciBertClassificationDescription
        case _:
            raise
