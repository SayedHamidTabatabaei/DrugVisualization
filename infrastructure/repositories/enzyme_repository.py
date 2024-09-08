from infrastructure.mysqldb.mysql_repository import MySqlRepository


class EnzymeRepository(MySqlRepository):
    def __init__(self):
        super().__init__()
        self.table_name = 'enzymes'

    def generate_enzyme_pivot(self, start, length):
        result, columns = self.call_procedure('GenerateEnzymePivot', [start, length])
        return result, columns

    def get_enzymes_by_drug_id(self, drug_id):
        result, _ = self.call_procedure('FindEnzymesByDrugId', [drug_id])
        return result
