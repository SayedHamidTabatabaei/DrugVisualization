from injector import inject

from businesses.base_business import BaseBusiness
from businesses.similarity_business import SimilarityBusiness
from common.enums.category import Category
from common.enums.similarity_type import SimilarityType
from infrastructure.repositories.drug_repository import DrugRepository
from infrastructure.repositories.similarity_repository import SimilarityRepository
from infrastructure.repositories.target_repository import TargetRepository


class TargetBusiness(BaseBusiness):
    @inject
    def __init__(self, similarity_business: SimilarityBusiness, drug_repository: DrugRepository,
                 target_repository: TargetRepository, similarity_repository: SimilarityRepository):
        BaseBusiness.__init__(self)
        self.similarity_business = similarity_business
        self.drug_repository = drug_repository
        self.target_repository = target_repository
        self.similarity_repository = similarity_repository

    def get_targets(self, drugbank_id):
        drug_id = self.drug_repository.get_id_by_drugbank_id(drugbank_id)

        targets = self.target_repository.get_targets_by_drug_id(drug_id)

        columns = ['id', 'target_code', 'target_name', 'position', 'organism']

        data = [dict(zip(columns, row)) for row in targets[0]]

        return data

    def get_drug_targets(self, start, length):
        total_number = self.drug_repository.get_active_drug_number()

        targets, columns = self.target_repository.generate_target_pivot(start, length)

        column_names = [item[0] for item in columns[0]]

        data = [dict(zip(column_names, row)) for row in targets[0]]

        return column_names, data, total_number

    def get_target_similarity(self, similarity_type: SimilarityType, start: int, length: int):

        columns, data, total_number = self.similarity_business.get_similarity_grid_data(
            similarity_type, Category.Target, True, True, True, True, start, length)

        return columns, data, total_number

    def generate_similarity(self, similarity_type: SimilarityType):
        results, columns_description = self.target_repository.generate_target_pivot(0, 100000)

        codes, values = results[1], results[0]

        self.similarity_business.calculate_similarity(codes, values, columns_description, similarity_type,
                                                      Category.Target)
