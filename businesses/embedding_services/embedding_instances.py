from businesses.embedding_services.bigbird_embedding_service import BigBirdEmbeddingService
from businesses.embedding_services.embedding_base_service import EmbeddingBaseService
from businesses.embedding_services.led_embedding_service import LedEmbeddingService
from businesses.embedding_services.longformer_embedding_service import LongFormerEmbeddingService
from businesses.embedding_services.pubmedbert_embedding_service import PubmedBertEmbeddingService
from businesses.embedding_services.scibert_embedding_service import SciBertEmbeddingService
from common.enums.embedding_type import EmbeddingType


def get_instance(category: EmbeddingType) -> EmbeddingBaseService:

    if category == EmbeddingType.PubMedBERT:
        return PubmedBertEmbeddingService(category)
    elif category == EmbeddingType.SciBERT:
        return SciBertEmbeddingService(category)
    elif category == EmbeddingType.LongFormer_BioNER:
        return LongFormerEmbeddingService(category)
    elif category == EmbeddingType.BigBird_PubMed:
        return BigBirdEmbeddingService(category)
    elif category == EmbeddingType.LED:
        return LedEmbeddingService(category)
    else:
        raise ValueError("No suitable subclass found")
