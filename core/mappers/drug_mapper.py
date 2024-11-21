from core.repository_models.drug_information_dto import DrugInformationDTO
from core.repository_models.drug_smiles_dto import DrugSmilesDTO
from core.repository_models.drug_text_property_dto import DrugTextPropertyDTO
from core.repository_models.training_drug_data_dto import TrainingDrugDataDTO


def map_drug_information_dto(query_result) -> DrugInformationDTO:
    (id, drugbank_drug_id, drug_name, drugbank_id, smiles, drug_type, description, average_mass, monoisotopic_mass, state, indication, pharmacodynamics,
     mechanism_of_action, toxicity, metabolism, absorption, half_life, protein_binding, route_of_elimination, volume_of_distribution, clearance,
     classification_description, total_text, classification_direct_parent, classification_kingdom, classification_superclass, classification_class_category,
     classification_subclass, bioavailability, ghose_filter, h_bond_acceptor_count, h_bond_donor_count, log_p, log_s, mddr_like_rule, molecular_formula,
     molecular_weight, monoisotopic_weight, number_of_rings, physiological_charge, pka_strongest_acidic, pka_strongest_basic, polar_surface_area,
     polarizability, refractivity, rotatable_bond_count, rule_of_five, water_solubility, rdkit_3d, rdkit_2d, has_enzyme, has_pathway, has_target,
     has_smiles) = query_result

    return DrugInformationDTO(id=id,
                              drugbank_drug_id=drugbank_drug_id,
                              drug_name=drug_name,
                              drugbank_id=drugbank_id,
                              smiles=smiles,
                              drug_type=drug_type,
                              description=description,
                              average_mass=average_mass,
                              monoisotopic_mass=monoisotopic_mass,
                              state=state,
                              indication=indication,
                              pharmacodynamics=pharmacodynamics,
                              mechanism_of_action=mechanism_of_action,
                              toxicity=toxicity,
                              metabolism=metabolism,
                              absorption=absorption,
                              half_life=half_life,
                              protein_binding=protein_binding,
                              route_of_elimination=route_of_elimination,
                              volume_of_distribution=volume_of_distribution,
                              clearance=clearance,
                              classification_description=classification_description,
                              total_text=total_text,
                              classification_direct_parent=classification_direct_parent,
                              classification_kingdom=classification_kingdom,
                              classification_superclass=classification_superclass,
                              classification_class_category=classification_class_category,
                              classification_subclass=classification_subclass,
                              bioavailability=bioavailability,
                              ghose_filter=ghose_filter,
                              h_bond_acceptor_count=h_bond_acceptor_count,
                              h_bond_donor_count=h_bond_donor_count,
                              log_p=log_p,
                              log_s=log_s,
                              mddr_like_rule=mddr_like_rule,
                              molecular_formula=molecular_formula,
                              molecular_weight=molecular_weight,
                              monoisotopic_weight=monoisotopic_weight,
                              number_of_rings=number_of_rings,
                              physiological_charge=physiological_charge,
                              pka_strongest_acidic=pka_strongest_acidic,
                              pka_strongest_basic=pka_strongest_basic,
                              polar_surface_area=polar_surface_area,
                              polarizability=polarizability,
                              refractivity=refractivity,
                              rotatable_bond_count=rotatable_bond_count,
                              rule_of_five=rule_of_five,
                              water_solubility=water_solubility,
                              rdkit_3d=rdkit_3d,
                              rdkit_2d=rdkit_2d,
                              has_enzyme=has_enzyme,
                              has_pathway=has_pathway,
                              has_target=has_target,
                              has_smiles=has_smiles)


def map_drug_smiles(query_results) -> list[DrugSmilesDTO]:
    drugs = []
    for result in query_results:
        drug_id, drugbank_id, smiles, has_enzyme, has_pathway, has_target, has_smiles = result
        drug = DrugSmilesDTO(id=drug_id,
                             drugbank_id=drugbank_id,
                             smiles=smiles,
                             has_enzyme=has_enzyme,
                             has_pathway=has_pathway,
                             has_target=has_target,
                             has_smiles=has_smiles,
                             fingerprint='')
        drugs.append(drug)

    return drugs


def map_drug_text(query_results) -> list[DrugTextPropertyDTO]:
    drugs = []
    for result in query_results:
        drug_id, text = result
        drug = DrugTextPropertyDTO(id=drug_id, text=text)
        drugs.append(drug)

    return drugs


def map_training_drug_data_dto(query_results) -> list[TrainingDrugDataDTO]:
    drugs = []
    for result in query_results:
        id, drug_name, drugbank_id = result
        drug = TrainingDrugDataDTO(drug_id=id, drug_name=drug_name, drugbank_id=drugbank_id)
        drugs.append(drug)

    return drugs
