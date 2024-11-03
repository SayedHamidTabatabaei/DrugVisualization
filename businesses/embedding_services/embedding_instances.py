from businesses.embedding_services.embedding_base_service import EmbeddingBaseService
from businesses.embedding_services.pubmedbert_embedding_service import PubmedBertEmbeddingService
from businesses.embedding_services.scibert_embedding_service import SciBertEmbeddingService
from common.enums.embedding_type import EmbeddingType


def get_instance(category: EmbeddingType) -> EmbeddingBaseService:

    match category:
        case EmbeddingType.PubMedBERT:
            return PubmedBertEmbeddingService(category)
        case EmbeddingType.SciBERT:
            return SciBertEmbeddingService(category)
        case EmbeddingType.PubMedBERT_32:
            return PubmedBertEmbeddingService(category, 32)
        case EmbeddingType.SciBERT_32:
            return SciBertEmbeddingService(category, 32)
        case EmbeddingType.PubMedBERT_64:
            return PubmedBertEmbeddingService(category, 64)
        case EmbeddingType.SciBERT_64:
            return SciBertEmbeddingService(category, 64)
        case EmbeddingType.PubMedBERT_128:
            return PubmedBertEmbeddingService(category, 128)
        case EmbeddingType.SciBERT_128:
            return SciBertEmbeddingService(category, 128)
        case EmbeddingType.PubMedBERT_256:
            return PubmedBertEmbeddingService(category, 256)
        case EmbeddingType.SciBERT_256:
            return SciBertEmbeddingService(category, 256)
        case EmbeddingType.PubMedBERT_512:
            return PubmedBertEmbeddingService(category, 512)
        case EmbeddingType.SciBERT_512:
            return SciBertEmbeddingService(category, 512)
        case _:
            raise ValueError("No suitable subclass found")
