from injector import inject

from businesses.base_business import BaseBusiness
from businesses.reduction_services import reduction_instances
from businesses.similarity_business import SimilarityBusiness
from common.enums.category import Category
from common.enums.embedding_type import EmbeddingType
from common.enums.reduction_category import ReductionCategory
from common.enums.similarity_type import SimilarityType
from common.enums.text_type import TextType
from common.helpers import embedding_helper
from core.models.reduction_parameter_model import ReductionParameterModel
from infrastructure.repositories.drug_embedding_repository import DrugEmbeddingRepository
from infrastructure.repositories.drug_repository import DrugRepository
from infrastructure.repositories.reduction_data_repository import ReductionDataRepository


class ReductionBusiness(BaseBusiness):
    @inject
    def __init__(self, reduction_data_repository: ReductionDataRepository,
                 drug_repository: DrugRepository,
                 drug_embedding_repository: DrugEmbeddingRepository, similarity_business: SimilarityBusiness):
        BaseBusiness.__init__(self)
        self.reduction_data_repository = reduction_data_repository
        self.drug_repository = drug_repository
        self.drug_embedding_repository = drug_embedding_repository
        self.similarity_business = similarity_business

    def get_reduction_embeddings(self, reduction_category: ReductionCategory, embedding_type: EmbeddingType,
                                 text_type: TextType, start: int, length: int):
        category = embedding_helper.find_category(embedding_type, text_type)

        total_number = self.reduction_data_repository.get_reduction_count(reduction_category, SimilarityType.Original,
                                                                          category, True, True, True, True)

        reduction_data = self.reduction_data_repository.find_reductions(reduction_category, SimilarityType.Original,
                                                                        category, True, True, True, True, start, length)

        return reduction_data, total_number

    def get_reduction_similarity(self, reduction_category: ReductionCategory, similarity_type: SimilarityType,
                                 category: Category, start: int, length: int):
        total_number = self.reduction_data_repository.get_reduction_count(reduction_category, similarity_type,
                                                                          category, True, True, True, True)

        reduction_data = self.reduction_data_repository.find_reductions(reduction_category, similarity_type,
                                                                        category, True, True, True, True, start, length)

        return reduction_data, total_number

    def calculate_reduction_similarity(self, reduction_category: ReductionCategory, similarity_type: SimilarityType,
                                       category: Category):
        if similarity_type != SimilarityType.Original:
            result = self.similarity_business.get_all_similarities(similarity_type=similarity_type, category=category,
                                                                   check_target=True, check_pathway=True,
                                                                   check_enzyme=True, check_smiles=True)
        else:
            result = self.get_original_data(category=category)

        instance = reduction_instances.get_instance(reduction_category)

        reductions = instance.reduce(ReductionParameterModel(similarity_type=similarity_type,
                                                             category=category,
                                                             reduction_category=reduction_category),
                                     result)

        self.reduction_data_repository.insert_batch_check_duplicate(reduction_datas=reductions)

    def calculate_reduction_embedding(self, reduction_category: ReductionCategory, text_type: TextType,
                                      embedding_type: EmbeddingType):
        embeddings = self.drug_embedding_repository.find_all_embedding_dict(embedding_type=embedding_type,
                                                                            text_type=text_type)

        embeddings = {key: [float(x) for x in values.replace('[', '').replace(']', '').split()]
                      for key, values in embeddings.items()}

        category = embedding_helper.find_category(embedding_type, text_type)

        instance = reduction_instances.get_instance(reduction_category)

        reductions = instance.reduce(ReductionParameterModel(similarity_type=SimilarityType.Original,
                                                             category=category,
                                                             reduction_category=reduction_category),
                                     embeddings)

        self.reduction_data_repository.insert_batch_check_duplicate(reduction_datas=reductions)

    def get_original_data(self, category: Category):
        match category:
            case Category.Substructure:
                results = self.drug_repository.get_all_drug_smiles()

                return {r.id: r.smiles for r in results}
            case Category.Target:
                pass
            case Category.Pathway:
                pass
            case Category.Enzyme:
                pass
            case _:
                raise Exception(f'Unexpected category {category}')
