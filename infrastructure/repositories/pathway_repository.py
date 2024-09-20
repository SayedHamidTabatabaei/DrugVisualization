from infrastructure.mysqldb.mysql_repository import MySqlRepository


class PathwayRepository(MySqlRepository):
    def __init__(self):
        super().__init__('pathways')

    def generate_pathway_pivot(self, start, length):
        result, columns = self.call_procedure('GeneratePathwayPivot', [start, length])

        return result, columns

    def get_pathways_by_drug_id(self, drug_id):
        result, _ = self.call_procedure('FindPathwaysByDrugId', [drug_id])
        return result
