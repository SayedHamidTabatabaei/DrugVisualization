from core.mappers.drug_interaction_mapper import map_training_drug_interactions
from core.mappers.drug_mapper import map_drug_text
from core.repository_models.drug_text_property_dto import DrugTextPropertyDTO
from core.repository_models.training_drug_interaction_dto import TrainingDrugInteractionDTO
from infrastructure.mysqldb.mysql_repository import MySqlRepository


class DrugInteractionRepository(MySqlRepository):
    def __init__(self):
        super().__init__('drug_interactions')

    def find_all_interaction_description(self) -> list[DrugTextPropertyDTO]:
        result, _ = self.call_procedure('FindAllDrugInteractionDescription')
        return map_drug_text(result[0])

    def find_training_interactions(self, has_enzyme: bool, has_pathway: bool, has_target: bool, has_smiles: bool) -> list[TrainingDrugInteractionDTO]:
        result, _ = self.call_procedure('FindInteractions',
                                        [has_enzyme, has_pathway, has_target, has_smiles])

        return map_training_drug_interactions(result[0])
