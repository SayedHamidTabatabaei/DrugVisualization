from transformers import AutoTokenizer, AutoModel

from common.enums.category import Category
from common.enums.embedding_type import EmbeddingType
from common.enums.text_type import TextType

pubmedbert_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                                                     clean_up_tokenization_spaces=False)
pubmedbert_model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")


def get_pubmedbert_embedding(text):
    issue_on_max_length = False

    try:
        inputs = pubmedbert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = pubmedbert_model(**inputs)
    except Exception as e:
        inputs = pubmedbert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = pubmedbert_model(**inputs)
        issue_on_max_length = True

    return outputs.last_hidden_state.mean(dim=1).detach().numpy(), issue_on_max_length
    pass


scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased",
                                                  clean_up_tokenization_spaces=False)
scibert_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")


def get_scibert_embedding(text):

    issue_on_max_length = False

    try:
        inputs = scibert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = scibert_model(**inputs)
    except Exception as e:
        inputs = scibert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = scibert_model(**inputs)
        issue_on_max_length = True

    return outputs.last_hidden_state.mean(dim=1).detach().numpy(), issue_on_max_length
    pass


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
