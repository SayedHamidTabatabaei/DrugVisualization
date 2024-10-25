import numpy as np
from tqdm import tqdm

from core.repository_models.training_drug_interaction_dto import TrainingDrugInteractionDTO


def map_training_drug_interactions(query_results) -> list[TrainingDrugInteractionDTO]:
    interactions = []

    for result in tqdm(query_results, "Mapping Interactions...."):
        (drug_1_id, drugbank_id_1, drug_2_id, drugbank_id_2, interaction_type) = result
        interaction_entity = TrainingDrugInteractionDTO(drug_1=drug_1_id,
                                                        drug_2=drug_2_id,
                                                        interaction_type=np.int16(interaction_type))
        interactions.append(interaction_entity)

    return interactions
