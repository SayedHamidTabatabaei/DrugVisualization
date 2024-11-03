from core.mappers import pathway_mapper
from infrastructure.mysqldb.mysql_repository import MySqlRepository


class PathwayRepository(MySqlRepository):
    def __init__(self):
        super().__init__('pathways')

    def get_drug_pathway_as_feature(self):
        pathways, _ = self.generate_pathway_pivot(start=0, length=100000, has_enzyme=True, has_target=True, has_smiles=True)

        return pathway_mapper.map_pathway_features(pathways[0])

    def generate_pathway_pivot(self, start, length, has_enzyme: bool = False, has_target: bool = False, has_smiles: bool = False):
        result, columns = self.call_procedure('GeneratePathwayPivot', [start, length, has_enzyme, has_target, has_smiles])

        return result, columns

    def get_pathways_by_drug_id(self, drug_id):
        result, _ = self.call_procedure('FindPathwaysByDrugId', [drug_id])
        return result
