from infrastructure.mysqldb.mysql_repository import MySqlRepository


class TargetRepository(MySqlRepository):
    def __init__(self):
        super().__init__()
        self.table_name = 'targets'

    def generate_target_pivot(self, start, length):
        result, columns = self.call_procedure('GenerateTargetPivot', [start, length])

        return result, columns

    def get_targets_by_drug_id(self, drug_id):
        result, _ = self.call_procedure('FindTargetsByDrugId', [drug_id])
        return result
