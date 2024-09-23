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
        case _:
            raise ValueError("No suitable subclass found")
