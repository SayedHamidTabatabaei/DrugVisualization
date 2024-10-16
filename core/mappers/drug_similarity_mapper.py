from tqdm import tqdm

from common.enums.category import Category
from common.enums.similarity_type import SimilarityType
from core.repository_models.drug_similarity_dto import DrugSimilarityDTO


def map_drug_similarity(query_results) -> list[DrugSimilarityDTO]:
    similarities = []
    for result in tqdm(query_results, 'Fetching Similarities....'):
        similarity_type, category, drug_1_id, drug_2_id, drugbank_id_1, drugbank_id_2, value = result
        similarity_entity = DrugSimilarityDTO(similarity_type=SimilarityType(similarity_type),
                                              category=Category(category),
                                              drug_1=drug_1_id,
                                              drug_2=drug_2_id,
                                              drugbank_id_1=drugbank_id_1,
                                              drugbank_id_2=drugbank_id_2,
                                              value=float(value))
        similarities.append(similarity_entity)

    return similarities
