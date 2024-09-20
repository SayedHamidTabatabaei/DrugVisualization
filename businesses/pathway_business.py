from injector import inject

from businesses.base_business import BaseBusiness
from businesses.similarity_business import SimilarityBusiness
from common.enums.category import Category
from common.enums.similarity_type import SimilarityType
from infrastructure.repositories.drug_repository import DrugRepository
from infrastructure.repositories.pathway_repository import PathwayRepository
from infrastructure.repositories.similarity_repository import SimilarityRepository


class PathwayBusiness(BaseBusiness):
    @inject
    def __init__(self, similarity_business: SimilarityBusiness, drug_repository: DrugRepository,
                 pathway_repository: PathwayRepository, similarity_repository: SimilarityRepository):
        BaseBusiness.__init__(self)
        self.similarity_business = similarity_business
        self.drug_repository = drug_repository
        self.pathway_repository = pathway_repository
        self.similarity_repository = similarity_repository

    def get_pathways(self, drugbank_id):
        drug_id = self.drug_repository.get_id_by_drugbank_id(drugbank_id)

        pathways = self.pathway_repository.get_pathways_by_drug_id(drug_id)

        columns = ['id', 'pathway_code', 'pathway_name']

        data = [dict(zip(columns, row)) for row in pathways[0]]

        return data

    def get_drug_pathways(self, start, length):
        total_number = self.drug_repository.get_active_drug_number()

        pathways, columns = self.pathway_repository.generate_pathway_pivot(start, length)

        column_names = [item[0] for item in columns[0]]

        data = [dict(zip(column_names, row)) for row in pathways[0]]

        return column_names, data, total_number

    def get_pathway_similarity(self, similarity_type: SimilarityType, start: int, length: int):

        columns, data, total_number = self.similarity_business.get_similarity_grid_data(
            similarity_type, Category.Pathway, True, True, True, True, start, length)
        
        return columns, data, total_number

    def generate_similarity(self, similarity_type: SimilarityType):
        results, columns_description = self.pathway_repository.generate_pathway_pivot(0, 100000)

        codes, values = results[1], results[0]

        self.similarity_business.calculate_similarity(codes, values, columns_description, similarity_type,
                                                      Category.Pathway)
