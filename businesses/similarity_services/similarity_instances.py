from businesses.similarity_services.cosine_similarity_service import CosineSimilarityService
from businesses.similarity_services.jacquard_similarity_service import JacquardSimilarityService
from businesses.similarity_services.similarity_base_service import SimilarityBaseService
from common.enums.similarity_type import SimilarityType


def get_instance(category: SimilarityType) -> SimilarityBaseService:
    match category:
        case SimilarityType.Original:
            pass
        case SimilarityType.Jacquard:
            return JacquardSimilarityService(category)
        case SimilarityType.Cosine:
            return CosineSimilarityService(category)
        case SimilarityType.Euclidean:
            pass
        case SimilarityType.Manhattan:
            pass
        case SimilarityType.Hamming:
            pass
        case SimilarityType.Pearson:
            pass
        case SimilarityType.Spearman:
            pass
        case SimilarityType.Mahalanobis:
            pass
        case SimilarityType.Dice:
            pass
        case SimilarityType.Tanimoto:
            pass
        case SimilarityType.Kullback:
            pass
        case SimilarityType.Bhattacharyya:
            pass
        case SimilarityType.Edit:
            pass
        case SimilarityType.DynamicTime:
            pass
        case SimilarityType.Hausdorff:
            pass
        case _:
            raise ValueError("No suitable subclass found")
