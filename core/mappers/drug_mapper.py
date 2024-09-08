from core.repository_models.drug_absorption_dto import DrugAbsorptionDTO
from core.repository_models.drug_classification_description_dto import DrugClassificationDescriptionDTO
from core.repository_models.drug_clearance_dto import DrugClearanceDTO
from core.repository_models.drug_description_dto import DrugDescriptionDTO
from core.repository_models.drug_half_life_dto import DrugHalfLifeDTO
from core.repository_models.drug_indication_dto import DrugIndicationDTO
from core.repository_models.drug_mechanism_of_action_dto import DrugMechanismOfActionDTO
from core.repository_models.drug_metabolism_dto import DrugMetabolismDTO
from core.repository_models.drug_pharmacodynamics_dto import DrugPharmacodynamicsDTO
from core.repository_models.drug_protein_binding_dto import DrugProteinBindingDTO
from core.repository_models.drug_route_of_elimination_dto import DrugRouteOfEliminationDTO
from core.repository_models.drug_smiles_dto import DrugSmilesDTO
from core.repository_models.drug_text_property_dto import DrugTextPropertyDTO
from core.repository_models.drug_toxicity_dto import DrugToxicityDTO
from core.repository_models.drug_volume_of_distribution_dto import DrugVolumeOfDistributionDTO


def map_drug_smiles(query_results) -> list[DrugSmilesDTO]:
    drugs = []
    for result in query_results:
        drug_id, smiles = result
        drug = DrugSmilesDTO(id=drug_id, smiles=smiles, fingerprint='')
        drugs.append(drug)

    return drugs


def map_drug_text(query_results) -> list[DrugTextPropertyDTO]:
    drugs = []
    for result in query_results:
        drug_id, text = result
        drug = DrugTextPropertyDTO(id=drug_id, text=text)
        drugs.append(drug)

    return drugs


def map_drug_description(query_results) -> list[DrugDescriptionDTO]:
    drugs = []
    for result in query_results:
        drug_id, description = result
        drug = DrugDescriptionDTO(id=drug_id, description=description)
        drugs.append(drug)

    return drugs


def map_drug_indication(query_results) -> list[DrugIndicationDTO]:
    drugs = []
    for result in query_results:
        drug_id, indication = result
        drug = DrugIndicationDTO(id=drug_id, indication=indication)
        drugs.append(drug)

    return drugs


def map_drug_pharmacodynamics(query_results) -> list[DrugPharmacodynamicsDTO]:
    drugs = []
    for result in query_results:
        drug_id, pharmacodynamics = result
        drug = DrugPharmacodynamicsDTO(id=drug_id, pharmacodynamics=pharmacodynamics)
        drugs.append(drug)

    return drugs


def map_drug_mechanism_of_action(query_results) -> list[DrugMechanismOfActionDTO]:
    drugs = []
    for result in query_results:
        drug_id, mechanism_of_action = result
        drug = DrugMechanismOfActionDTO(id=drug_id, mechanism_of_action=mechanism_of_action)
        drugs.append(drug)

    return drugs


def map_drug_toxicity(query_results) -> list[DrugToxicityDTO]:
    drugs = []
    for result in query_results:
        drug_id, toxicity = result
        drug = DrugToxicityDTO(id=drug_id, toxicity=toxicity)
        drugs.append(drug)

    return drugs


def map_drug_metabolism(query_results) -> list[DrugMetabolismDTO]:
    drugs = []
    for result in query_results:
        drug_id, metabolism = result
        drug = DrugMetabolismDTO(id=drug_id, metabolism=metabolism)
        drugs.append(drug)

    return drugs


def map_drug_absorption(query_results) -> list[DrugAbsorptionDTO]:
    drugs = []
    for result in query_results:
        drug_id, absorption = result
        drug = DrugAbsorptionDTO(id=drug_id, absorption=absorption)
        drugs.append(drug)

    return drugs


def map_drug_half_life(query_results) -> list[DrugHalfLifeDTO]:
    drugs = []
    for result in query_results:
        drug_id, half_life = result
        drug = DrugHalfLifeDTO(id=drug_id, half_life=half_life)
        drugs.append(drug)

    return drugs


def map_drug_protein_binding(query_results) -> list[DrugProteinBindingDTO]:
    drugs = []
    for result in query_results:
        drug_id, protein_binding = result
        drug = DrugProteinBindingDTO(id=drug_id, protein_binding=protein_binding)
        drugs.append(drug)

    return drugs


def map_drug_route_of_elimination(query_results) -> list[DrugRouteOfEliminationDTO]:
    drugs = []
    for result in query_results:
        drug_id, route_of_elimination = result
        drug = DrugRouteOfEliminationDTO(id=drug_id, route_of_elimination=route_of_elimination)
        drugs.append(drug)

    return drugs


def map_drug_volume_of_distribution(query_results) -> list[DrugVolumeOfDistributionDTO]:
    drugs = []
    for result in query_results:
        drug_id, volume_of_distribution = result
        drug = DrugVolumeOfDistributionDTO(id=drug_id, volume_of_distribution=volume_of_distribution)
        drugs.append(drug)

    return drugs


def map_drug_clearance(query_results) -> list[DrugClearanceDTO]:
    drugs = []
    for result in query_results:
        drug_id, clearance = result
        drug = DrugClearanceDTO(id=drug_id, clearance=clearance)
        drugs.append(drug)

    return drugs


def map_drug_classification_description(query_results) -> list[DrugClassificationDescriptionDTO]:
    drugs = []
    for result in query_results:
        drug_id, classification_description = result
        drug = DrugClassificationDescriptionDTO(id=drug_id, classification_description=classification_description)
        drugs.append(drug)

    return drugs
