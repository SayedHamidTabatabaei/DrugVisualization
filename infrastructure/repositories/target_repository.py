from core.mappers import target_mapper
from infrastructure.mysqldb.mysql_repository import MySqlRepository


class TargetRepository(MySqlRepository):
    def __init__(self):
        super().__init__('targets')

    def get_drug_target_as_feature(self):
        targets, _ = self.generate_target_pivot(start=0, length=100000, has_pathway=True, has_enzyme=True, has_smiles=True)

        return target_mapper.map_target_features(targets[0])

    def generate_target_pivot(self, start, length, has_pathway: bool = False, has_enzyme: bool = False, has_smiles: bool = False):
        result, columns = self.call_procedure('GenerateTargetPivot', [start, length, has_pathway, has_enzyme, has_smiles])

        return result, columns

    def get_targets_by_drug_id(self, drug_id):
        result, _ = self.call_procedure('FindTargetsByDrugId', [drug_id])
        return result
