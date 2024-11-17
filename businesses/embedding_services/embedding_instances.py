from businesses.embedding_services.embedding_base_service import EmbeddingBaseService
from businesses.embedding_services.pubmedbert_embedding_service import PubmedBertEmbeddingService
from businesses.embedding_services.scibert_embedding_service import SciBertEmbeddingService
from common.enums.embedding_type import EmbeddingType


def get_instance(category: EmbeddingType) -> EmbeddingBaseService:

    if category == EmbeddingType.PubMedBERT:
        return PubmedBertEmbeddingService(category)
    elif category == EmbeddingType.SciBERT:
        return SciBertEmbeddingService(category)
    elif category == EmbeddingType.PubMedBERT_32:
        return PubmedBertEmbeddingService(category, 32)
    elif category == EmbeddingType.SciBERT_32:
        return SciBertEmbeddingService(category, 32)
    elif category == EmbeddingType.PubMedBERT_64:
        return PubmedBertEmbeddingService(category, 64)
    elif category == EmbeddingType.SciBERT_64:
        return SciBertEmbeddingService(category, 64)
    elif category == EmbeddingType.PubMedBERT_128:
        return PubmedBertEmbeddingService(category, 128)
    elif category == EmbeddingType.SciBERT_128:
        return SciBertEmbeddingService(category, 128)
    elif category == EmbeddingType.PubMedBERT_256:
        return PubmedBertEmbeddingService(category, 256)
    elif category == EmbeddingType.SciBERT_256:
        return SciBertEmbeddingService(category, 256)
    elif category == EmbeddingType.PubMedBERT_512:
        return PubmedBertEmbeddingService(category, 512)
    elif category == EmbeddingType.SciBERT_512:
        return SciBertEmbeddingService(category, 512)
    else:
        raise ValueError("No suitable subclass found")
