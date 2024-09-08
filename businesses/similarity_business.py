from collections import defaultdict

from injector import inject
from tqdm import tqdm

from businesses.base_business import BaseBusiness
from common.enums.category import Category
from common.enums.similarity_type import SimilarityType
from common.helpers import matrix_helper, similarity_helper
from core.domain.similarity import Similarity
from infrastructure.repositories.reduction_data_repository import ReductionDataRepository
from infrastructure.repositories.similarity_repository import SimilarityRepository


class SimilarityBusiness(BaseBusiness):
    @inject
    def __init__(self, similarity_repository: SimilarityRepository, final_data_repository: ReductionDataRepository):
        BaseBusiness.__init__(self)
        self.similarity_repository = similarity_repository
        self.final_data_repository = final_data_repository

    def get_similarity_grid_data(self, similarity_type: SimilarityType, category: Category,
                                 check_target: bool, check_pathway: bool, check_enzyme: bool, check_smiles: bool,
                                 start: int, length: int):
        total_number = self.get_similarity_count(similarity_type, category,
                                                 check_target, check_pathway, check_enzyme, check_smiles)

        targets, columns = self.get_similarity_data(similarity_type, category,
                                                    check_target, check_pathway, check_enzyme, check_smiles,
                                                    start, length)

        data = [dict(zip(columns, row)) for row in targets]

        return columns, data, total_number

    def get_similarity_count(self, similarity_type: SimilarityType, category: Category,
                             check_target: bool, check_pathway: bool, check_enzyme: bool, check_smiles: bool):
        total_number = self.similarity_repository.get_similarity_count(similarity_type, category,
                                                                       check_target, check_pathway, check_enzyme,
                                                                       check_smiles)

        return total_number

    def get_similarity_data(self, similarity_type: SimilarityType, category: Category,
                            check_target: bool, check_pathway: bool, check_enzyme: bool, check_smiles: bool,
                            start: int, length: int):

        similarities = self.similarity_repository.find_all_similarity(similarity_type, category,
                                                                      check_target, check_pathway, check_enzyme,
                                                                      check_smiles, start, length)

        similarities = sorted(similarities, key=lambda d: (d.drug_1, d.drug_2))

        unique_drugs_1 = sorted({similarity.drug_1 for similarity in similarities})
        unique_drugs_2 = sorted({similarity.drug_2 for similarity in similarities})

        matrix = [[0] * (len(unique_drugs_2) + 1) for _ in range(len(unique_drugs_1))]

        columns = [''] * (len(unique_drugs_2) + 1)
        columns[0] = 'DrugBankId'

        for similarity in similarities:
            i = unique_drugs_1.index(similarity.drug_1)
            j = unique_drugs_2.index(similarity.drug_2)

            matrix[i][j + 1] = similarity.value

            if columns[j + 1] == '':
                columns[j + 1] = similarity.drugbank_id_2

            if matrix[i][0] == 0:
                matrix[i][0] = similarity.drugbank_id_1

        return matrix, columns

    def get_all_similarities(self, similarity_type: SimilarityType, category: Category,
                             check_target: bool, check_pathway: bool, check_enzyme: bool, check_smiles: bool):
        similarities = self.similarity_repository.find_all_similarity(similarity_type, category,
                                                                      check_target, check_pathway, check_enzyme,
                                                                      check_smiles, 0, 100000)

        similarities = sorted(similarities, key=lambda d: (d.drug_1, d.drug_2))

        aggregated_values = defaultdict(list)

        for similarity in similarities:
            key = similarity.drug_1
            aggregated_values[key].append(similarity.value)

        return aggregated_values

    def generate_jacquard(self, results, columns_description, category):

        codes = [item[0] for item in results[1]]
        values = results[0]

        features = matrix_helper.create_specific_matrix_by_column_names(values, codes, columns_description[0])

        jacquard = similarity_helper.generate_jacquard_similarity(features)

        self.insert_similarities(jacquard, values, SimilarityType.Jacquard, category)

    def generate_cosine(self, results, columns_description, category):

        codes = [item[0] for item in results[1]]
        values = results[0]

        features = matrix_helper.create_specific_matrix_by_column_names(values, codes, columns_description[0])

        cosine = similarity_helper.generate_cosine_similarity(features)

        self.insert_similarities(cosine, values, SimilarityType.Cosine, category)

    def insert_similarities(self, values, items, similarity_type: SimilarityType, category: Category):

        drug_ids = [item[0] for item in items]

        similarities: list[Similarity] = []

        for i in tqdm(range(len(drug_ids)), desc="Processing Insert similarity"):
            for j in range(len(drug_ids)):
                similarities.append(Similarity(similarity_type=similarity_type,
                                               category=category,
                                               drug_1=drug_ids[i],
                                               drug_2=drug_ids[j],
                                               value=values[i][j]))

        self.similarity_repository.insert_batch_check_duplicate(similarities)
