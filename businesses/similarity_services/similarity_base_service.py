from core.domain.similarity import Similarity
from core.repository_models.drug_smiles_dto import DrugSmilesDTO


class SimilarityBaseService:
    def __init__(self, category):
        self.category = category

    def calculate_similarity(self, codes, values, columns_description):
        pass

    def calculate_similes_similarity(self, all_drugs: list[DrugSmilesDTO]) -> list[Similarity]:
        pass

    @staticmethod
    def create_specific_matrix_by_column_names(base_matrix, column_codes, columns_description):
        columns = [desc[0] for desc in columns_description]

        filtered_columns = [col for col in columns if col in column_codes]

        matrix = []
        for row in base_matrix:
            filtered_row = [float(row[columns.index(col)]) for col in filtered_columns]
            matrix.append(filtered_row)

        return matrix
