from businesses.similarity_services.cosine_similarity_service import CosineSimilarityService
from businesses.similarity_services.drug_structure_similarity_service import DrugStructureSimilarityService
from businesses.similarity_services.jacquard_similarity_service import JacquardSimilarityService
from businesses.similarity_services.similarity_base_service import SimilarityBaseService
from common.enums.similarity_type import SimilarityType


def get_instance(category: SimilarityType) -> SimilarityBaseService:
    if category == SimilarityType.Original:
        pass
    elif category == SimilarityType.Jacquard:
        return JacquardSimilarityService(category)
    elif category == SimilarityType.Cosine:
        return CosineSimilarityService(category)
    elif category == SimilarityType.Euclidean:
        pass
    elif category == SimilarityType.Manhattan:
        pass
    elif category == SimilarityType.Hamming:
        pass
    elif category == SimilarityType.Pearson:
        pass
    elif category == SimilarityType.Spearman:
        pass
    elif category == SimilarityType.Mahalanobis:
        pass
    elif category == SimilarityType.Dice:
        pass
    elif category == SimilarityType.Tanimoto:
        pass
    elif category == SimilarityType.Kullback:
        pass
    elif category == SimilarityType.Bhattacharyya:
        pass
    elif category == SimilarityType.Edit:
        pass
    elif category == SimilarityType.DynamicTime:
        pass
    elif category == SimilarityType.Hausdorff:
        pass
    elif category == SimilarityType.DeepDDISmiles:
        return DrugStructureSimilarityService(category)
    else:
        raise ValueError("No suitable subclass found")
