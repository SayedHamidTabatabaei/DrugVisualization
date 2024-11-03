from core.mappers import enzyme_mapper
from infrastructure.mysqldb.mysql_repository import MySqlRepository


class EnzymeRepository(MySqlRepository):
    def __init__(self):
        super().__init__('enzymes')

    def get_drug_enzyme_as_feature(self):
        enzymes, _ = self.generate_enzyme_pivot(start=0, length=100000, has_pathway=True, has_target=True, has_smiles=True)

        return enzyme_mapper.map_enzyme_features(enzymes[0])

    def generate_enzyme_pivot(self, start, length, has_pathway: bool = False, has_target: bool = False, has_smiles: bool = False):
        result, columns = self.call_procedure('GenerateEnzymePivot', [start, length, has_pathway, has_target, has_smiles])
        return result, columns

    def get_enzymes_by_drug_id(self, drug_id):
        result, _ = self.call_procedure('FindEnzymesByDrugId', [drug_id])
        return result
