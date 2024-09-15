from injector import inject

from businesses.base_business import BaseBusiness
from businesses.similarity_business import SimilarityBusiness
from common.enums.category import Category
from common.enums.similarity_type import SimilarityType
from infrastructure.repositories.drug_repository import DrugRepository
from infrastructure.repositories.enzyme_repository import EnzymeRepository
from infrastructure.repositories.similarity_repository import SimilarityRepository


class EnzymeBusiness(BaseBusiness):
    @inject
    def __init__(self, similarity_business: SimilarityBusiness, drug_repository: DrugRepository,
                 enzyme_repository: EnzymeRepository, similarity_repository: SimilarityRepository):
        BaseBusiness.__init__(self)
        self.similarity_business = similarity_business
        self.drug_repository = drug_repository
        self.enzyme_repository = enzyme_repository
        self.similarity_repository = similarity_repository

    def get_enzymes(self, drugbank_id):
        drug_id = self.drug_repository.get_id_by_drugbank_id(drugbank_id)

        enzymes = self.enzyme_repository.get_enzymes_by_drug_id(drug_id)

        columns = ['id', 'enzyme_code', 'enzyme_name', 'position', 'organism']

        data = [dict(zip(columns, row)) for row in enzymes[0]]

        return data

    def get_drug_enzymes(self, start, length):
        total_number = self.drug_repository.get_active_drug_number()

        enzymes, columns = self.enzyme_repository.generate_enzyme_pivot(start, length)

        column_names = [item[0] for item in columns[0]]

        data = [dict(zip(column_names, row)) for row in enzymes[0]]

        return column_names, data, total_number

    def get_enzyme_similarity(self, similarity_type: SimilarityType, start: int, length: int):
        columns, data, total_number = self.similarity_business.get_similarity_grid_data(
            similarity_type, Category.Enzyme, True, True, True, True, start, length)

        return columns, data, total_number

    def generate_similarity(self, similarity_type: SimilarityType):
        results, columns_description = self.enzyme_repository.generate_enzyme_pivot(0, 100000)

        if similarity_type == SimilarityType.Jacquard:
            self.similarity_business.generate_jacquard(results, columns_description, Category.Enzyme)
        elif similarity_type == SimilarityType.Cosine:
            self.similarity_business.generate_cosine(results, columns_description, Category.Enzyme)
