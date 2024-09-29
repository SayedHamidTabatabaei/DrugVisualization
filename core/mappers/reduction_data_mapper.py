from tqdm import tqdm

from core.repository_models.reduction_data_dto import ReductionDataDTO
from core.repository_models.training_data_dto import TrainingDataDTO
from core.repository_models.interaction_dto import InteractionDTO


def map_training_data(query_results) -> list[TrainingDataDTO]:
    training_data = []

    for result in tqdm(query_results, desc="Mapping ..."):
        (drug_1_id, drugbank_id_1, reduction_values_1, drug_2_id,
         drugbank_id_2, reduction_values_2, interaction_type) = result
        training_entity = TrainingDataDTO(drug_1=drug_1_id,
                                          drugbank_id_1=drugbank_id_1,
                                          reduction_values_1=[float(val) for val in reduction_values_1[1:-1].split(',')],
                                          drug_2=drug_2_id,
                                          drugbank_id_2=drugbank_id_2,
                                          reduction_values_2=[float(val) for val in reduction_values_2[1:-1].split(',')],
                                          interaction_type=interaction_type)
        training_data.append(training_entity)

    return training_data


def map_interactions(query_results) -> list[InteractionDTO]:
    interactions = []

    for result in query_results:
        (drug_1_id, drugbank_id_1, drug_2_id,
         drugbank_id_2, interaction_type) = result
        interaction_entity = InteractionDTO(drug_1=drug_1_id,
                                            drugbank_id_1=drugbank_id_1,
                                            drug_2=drug_2_id,
                                            drugbank_id_2=drugbank_id_2,
                                            interaction_type=interaction_type)
        interactions.append(interaction_entity)

    return interactions


def map_reduction_data(query_results) -> list[ReductionDataDTO]:
    reduction_data = []
    for result in query_results:
        (id, drug_id, drugbank_id, similarity_type, category, reduction_category, reduction_value,
         has_enzyme, has_pathway, has_target, has_smiles) = result
        reduction_entity = ReductionDataDTO(id=id,
                                            drug_id=drug_id,
                                            drugbank_id=drugbank_id,
                                            similarity_type=similarity_type,
                                            category=category,
                                            reduction_category=reduction_category,
                                            reduction_value=reduction_value,
                                            has_enzyme=has_enzyme,
                                            has_pathway=has_pathway,
                                            has_target=has_target,
                                            has_smiles=has_smiles)
        reduction_data.append(reduction_entity)

    return reduction_data
