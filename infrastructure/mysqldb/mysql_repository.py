import mysql.connector
from tqdm import tqdm

from common.helpers import sql_helper
from configs.config import mysql_host, mysql_user, mysql_password, mysql_database_name, batch_size
from core.domain.base_model import BaseModel


class MySqlRepository:
    def __init__(self, table_name):
        self.host = mysql_host
        self.user = mysql_user
        self.password = mysql_password
        self.database = mysql_database_name
        self.batch_size = batch_size
        self.table_name = table_name

        try:
            self.connection = mysql.connector.connect(
                host=mysql_host,
                user=mysql_user,
                password=mysql_password,
                database=mysql_database_name,
                auth_plugin='mysql_native_password'
            )
        except mysql.connector.Error as err:
            raise Exception(err)

        self.cursor = self.connection.cursor()

    def insert(self, entity: BaseModel):
        query, values = sql_helper.generate_insert_command(entity)

        self.cursor.execute(query, values)

        self.connection.commit()

        return self.cursor.lastrowid

    def update(self, entity: BaseModel, update_columns: list):
        query, values = sql_helper.generate_update_command(entity, update_columns)

        self.cursor.execute(query, values)

        self.connection.commit()

        return self.cursor.rowcount

    def delete(self, id: int):
        query = sql_helper.generate_delete_command(self.table_name, id)

        self.cursor.execute(query)

        self.connection.commit()

        return self.cursor.rowcount

    def insert_batch(self, entities: list[BaseModel]):
        for i in tqdm(range(0, len(entities), batch_size), desc="Inserting data"):
            batch = entities[i:i + batch_size]
            query, values = sql_helper.generate_batch_insert_command(batch)

            self.cursor.executemany(query, values)

            self.connection.commit()

    def insert_batch_check_duplicate(self, entities: list[BaseModel], update_properties):
        # for i in tqdm(range(0, len(entities), batch_size), desc="Inserting data"):
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            query, values = sql_helper.generate_batch_insert_check_duplicate_command(batch, update_properties)

            self.cursor.executemany(query, values)

            self.connection.commit()

    def execute_query(self, query, params=None):
        self.cursor.execute(query, params)
        self.connection.commit()

    def execute_many_query(self, query, params=None):
        self.cursor.executemany(query, params)
        self.connection.commit()

    def call_procedure(self, procedure_name, args=()):
        self.cursor.callproc(procedure_name, args)

        results = []
        columns = []
        for result in self.cursor.stored_results():
            results.append(result.fetchall())
            columns.append(result.description)

        self.connection.commit()
        return results, columns

    def call_procedure_by_paging(self, procedure_name, start, length, args=()):

        if not args:
            self.cursor.callproc(procedure_name, [start, length])
        else:
            self.cursor.callproc(procedure_name, [start, length, args])

        results = []
        columns = []
        for result in self.cursor.stored_results():
            results.append(result.fetchall())
            columns.append(result.description)

        self.connection.commit()
        return results, columns

    def execute_query_and_return_last_row_id(self, query, params=None):
        self.cursor.execute(query, params)
        self.connection.commit()
        return self.cursor.lastrowid

    def execute_many_queries_and_return_last_row_id(self, query, params=None):
        self.cursor.executemany(query, params)
        self.connection.commit()
        return self.cursor.lastrowid

    def fetch_query(self, query, params=None):
        self.cursor.execute(query, params)
        return self.cursor.fetchall()

    def commit_connection(self):
        self.connection.commit()

    def close_connection(self):
        self.cursor.close()
        self.connection.close()
