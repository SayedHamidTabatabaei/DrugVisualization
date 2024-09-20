from core.mappers.drug_mapper import map_drug_smiles, map_drug_text
from core.repository_models.drug_smiles_dto import DrugSmilesDTO
from core.repository_models.drug_text_property_dto import DrugTextPropertyDTO
from infrastructure.mysqldb.mysql_repository import MySqlRepository


class DrugRepository(MySqlRepository):
    def __init__(self):
        super().__init__('drugs')

    def get_active_drug_number(self):
        result, _ = self.call_procedure('GetActiveDrugNumber')
        return result

    def get_list(self):
        result, _ = self.call_procedure('FindAllDrugs')
        return result

    def get_id_by_drugbank_id(self, drugbank_id):
        result, _ = self.call_procedure('FindIdByDrugBankId', [drugbank_id])
        return result[0][0][0]

    def find_smiles_by_drugbankid(self, drugbank_id):
        result, _ = self.call_procedure('FindSmilesByDrugBankId', [drugbank_id])
        return result

    def find_rdkit_by_drugbankid(self, drugbank_id):
        result, _ = self.call_procedure('FindRdkitByDrugBankId', [drugbank_id])
        return result

    def get_drug_information_by_drugbank_id(self, drugbank_id):
        result, _ = self.call_procedure('FindDrugInformationByDrugBankId', [drugbank_id])
        return result

    def get_interactions_by_drug_id(self, drug_id):
        result, _ = self.call_procedure('FindInteractionsByDrugId', [drug_id])
        return result

    def get_all_drug_smiles(self) -> list[DrugSmilesDTO]:
        result, _ = self.call_procedure('FindAllDrugSmiles')
        return map_drug_smiles(result[0])

    def find_all_drug_description(self) -> list[DrugTextPropertyDTO]:
        result, _ = self.call_procedure('FindAllDrugDescription')
        return map_drug_text(result[0])

    def find_all_drug_indication(self) -> list[DrugTextPropertyDTO]:
        result, _ = self.call_procedure('FindAllDrugIndication')
        return map_drug_text(result[0])

    def find_all_drug_pharmacodynamics(self) -> list[DrugTextPropertyDTO]:
        result, _ = self.call_procedure('FindAllDrugPharmacodynamics')
        return map_drug_text(result[0])

    def find_all_drug_mechanism_of_action(self) -> list[DrugTextPropertyDTO]:
        result, _ = self.call_procedure('FindAllDrugMechanismOfAction')
        return map_drug_text(result[0])

    def find_all_drug_toxicity(self) -> list[DrugTextPropertyDTO]:
        result, _ = self.call_procedure('FindAllDrugToxicity')
        return map_drug_text(result[0])

    def find_all_drug_metabolism(self) -> list[DrugTextPropertyDTO]:
        result, _ = self.call_procedure('FindAllDrugMetabolism')
        return map_drug_text(result[0])

    def find_all_drug_absorption(self) -> list[DrugTextPropertyDTO]:
        result, _ = self.call_procedure('FindAllDrugAbsorption')
        return map_drug_text(result[0])

    def find_all_drug_half_life(self) -> list[DrugTextPropertyDTO]:
        result, _ = self.call_procedure('FindAllDrugHalfLife')
        return map_drug_text(result[0])

    def find_all_drug_protein_binding(self) -> list[DrugTextPropertyDTO]:
        result, _ = self.call_procedure('FindAllDrugProteinBinding')
        return map_drug_text(result[0])

    def find_all_drug_route_of_elimination(self) -> list[DrugTextPropertyDTO]:
        result, _ = self.call_procedure('FindAllDrugRouteOfElimination')
        return map_drug_text(result[0])

    def find_all_drug_volume_of_distribution(self) -> list[DrugTextPropertyDTO]:
        result, _ = self.call_procedure('FindAllDrugVolumeOfDistribution')
        return map_drug_text(result[0])

    def find_all_drug_clearance(self) -> list[DrugTextPropertyDTO]:
        result, _ = self.call_procedure('FindAllDrugClearance')
        return map_drug_text(result[0])

    def find_all_drug_classification_description(self) -> list[DrugTextPropertyDTO]:
        result, _ = self.call_procedure('FindAllDrugClassificationDescription')
        return map_drug_text(result[0])
