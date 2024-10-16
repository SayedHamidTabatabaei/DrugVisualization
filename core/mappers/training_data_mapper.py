from tqdm import tqdm

from common.enums.category import Category
from common.enums.reduction_category import ReductionCategory
from common.enums.similarity_type import SimilarityType
from core.repository_models.interaction_dto import InteractionDTO
from core.repository_models.reduction_data_dto import ReductionDataDTO
from core.repository_models.training_data_dto import TrainingInteractionDataDTO


def map_interaction_similarities_training_data(interactions: list[InteractionDTO], reductions: list[ReductionDataDTO], category: Category) -> (
        list)[TrainingInteractionDataDTO]:
    if (SimilarityType(reductions[0].similarity_type) == SimilarityType.Original and
            Category(reductions[0].category) == Category.Substructure and
            ReductionCategory(reductions[0].reduction_category) == ReductionCategory.OriginalData):
        reduction_dict = {reduction.drug_id: [reduction.reduction_value] for reduction in reductions}
    else:
        reduction_dict = {reduction.drug_id: [float(val) for val in reduction.reduction_value[1:-1].split(',')]
                          for reduction in reductions}

    training_data = []

    for interaction in tqdm(interactions, 'Mapping training data', mininterval=0.5):
        training_entity = TrainingInteractionDataDTO(drug_1=interaction.drug_1,
                                                     drugbank_id_1=interaction.drugbank_id_1,
                                                     reduction_values_1=reduction_dict[interaction.drug_1],
                                                     drug_2=interaction.drug_2,
                                                     drugbank_id_2=interaction.drugbank_id_2,
                                                     reduction_values_2=reduction_dict[interaction.drug_2],
                                                     category=category,
                                                     interaction_type=interaction.interaction_type)
        training_data.append(training_entity)

    return training_data
