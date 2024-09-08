from injector import inject

from businesses.base_business import BaseBusiness
from businesses.similarity_business import SimilarityBusiness
from common.enums.category import Category
from common.enums.embedding_type import EmbeddingType
from common.enums.reduction_category import ReductionCategory
from common.enums.similarity_type import SimilarityType
from common.enums.text_type import TextType
from core.domain.reduction_data import ReductionData
from infrastructure.repositories.drug_embedding_repository import DrugEmbeddingRepository
from infrastructure.repositories.reduction_data_repository import ReductionDataRepository


def find_category(embedding_type: EmbeddingType, text_type: TextType) -> Category:
    match (embedding_type, text_type):
        case (EmbeddingType.PubMedBERT, TextType.Description):
            return Category.PubmedBertDescription
        case (EmbeddingType.PubMedBERT, TextType.Indication):
            return Category.PubmedBertIndication
        case (EmbeddingType.PubMedBERT, TextType.Pharmacodynamics):
            return Category.PubmedBertPharmacodynamics
        case (EmbeddingType.PubMedBERT, TextType.MechanismOfAction):
            return Category.PubmedBertMechanismOfAction
        case (EmbeddingType.PubMedBERT, TextType.Toxicity):
            return Category.PubmedBertToxicity
        case (EmbeddingType.PubMedBERT, TextType.Metabolism):
            return Category.PubmedBertMetabolism
        case (EmbeddingType.PubMedBERT, TextType.Absorption):
            return Category.PubmedBertAbsorption
        case (EmbeddingType.PubMedBERT, TextType.HalfLife):
            return Category.PubmedBertHalfLife
        case (EmbeddingType.PubMedBERT, TextType.ProteinBinding):
            return Category.PubmedBertProteinBinding
        case (EmbeddingType.PubMedBERT, TextType.RouteOfElimination):
            return Category.PubmedBertRouteOfElimination
        case (EmbeddingType.PubMedBERT, TextType.VolumeOfDistribution):
            return Category.PubmedBertVolumeOfDistribution
        case (EmbeddingType.PubMedBERT, TextType.Clearance):
            return Category.PubmedBertClearance
        case (EmbeddingType.PubMedBERT, TextType.ClassificationDescription):
            return Category.PubmedBertClassificationDescription
        case (EmbeddingType.SciBERT, TextType.Description):
            return Category.SciBertDescription
        case (EmbeddingType.SciBERT, TextType.Indication):
            return Category.SciBertIndication
        case (EmbeddingType.SciBERT, TextType.Pharmacodynamics):
            return Category.SciBertPharmacodynamics
        case (EmbeddingType.SciBERT, TextType.MechanismOfAction):
            return Category.SciBertMechanismOfAction
        case (EmbeddingType.SciBERT, TextType.Toxicity):
            return Category.SciBertToxicity
        case (EmbeddingType.SciBERT, TextType.Metabolism):
            return Category.SciBertMetabolism
        case (EmbeddingType.SciBERT, TextType.Absorption):
            return Category.SciBertAbsorption
        case (EmbeddingType.SciBERT, TextType.HalfLife):
            return Category.SciBertHalfLife
        case (EmbeddingType.SciBERT, TextType.ProteinBinding):
            return Category.SciBertProteinBinding
        case (EmbeddingType.SciBERT, TextType.RouteOfElimination):
            return Category.SciBertRouteOfElimination
        case (EmbeddingType.SciBERT, TextType.VolumeOfDistribution):
            return Category.SciBertVolumeOfDistribution
        case (EmbeddingType.SciBERT, TextType.Clearance):
            return Category.SciBertClearance
        case (EmbeddingType.SciBERT, TextType.ClassificationDescription):
            return Category.SciBertClassificationDescription
        case _:
            raise


class ReductionBusiness(BaseBusiness):
    @inject
    def __init__(self, reduction_data_repository: ReductionDataRepository,
                 drug_embedding_repository: DrugEmbeddingRepository, similarity_business: SimilarityBusiness):
        BaseBusiness.__init__(self)
        self.reduction_data_repository = reduction_data_repository
        self.drug_embedding_repository = drug_embedding_repository
        self.similarity_business = similarity_business

    def get_reduction_embeddings(self, reduction_category: ReductionCategory, embedding_type: EmbeddingType,
                                 text_type: TextType, start: int, length: int):
        category = find_category(embedding_type, text_type)

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

        result = self.similarity_business.get_all_similarities(similarity_type=similarity_type, category=category,
                                                               check_target=True, check_pathway=True,
                                                               check_enzyme=True, check_smiles=True)

        if reduction_category == ReductionCategory.OriginalData:
            self.calculate_original_data(result, similarity_type, Category.Target)
        elif reduction_category == ReductionCategory.AutoEncoder:
            pass

    def calculate_original_data(self, data, similarity_type: SimilarityType, category: Category):

        reductions = [ReductionData(drug_id=key,
                                    similarity_type=similarity_type,
                                    category=category,
                                    reduction_category=ReductionCategory.OriginalData,
                                    reduction_values=str(values),
                                    has_enzyme=True,
                                    has_pathway=True,
                                    has_target=True,
                                    has_smiles=True) for key, values in data.items()]

        self.reduction_data_repository.insert_batch_check_duplicate(reduction_datas=reductions)

    def calculate_reduction_embedding(self, reduction_category: ReductionCategory, text_type: TextType,
                                      embedding_type: EmbeddingType):
        embeddings = self.drug_embedding_repository.find_all_embedding(embedding_type=embedding_type,
                                                                       text_type=text_type)

        category = find_category(embedding_type, text_type)

        if reduction_category == ReductionCategory.OriginalData:
            self.calculate_embedding_original_data(embeddings, SimilarityType.Original, category)
        elif reduction_category == ReductionCategory.AutoEncoder:
            pass

    def calculate_embedding_original_data(self, embeddings, similarity_type: SimilarityType, category: Category):

        reductions = [ReductionData(drug_id=item.drug_id,
                                    similarity_type=similarity_type,
                                    category=category,
                                    reduction_category=ReductionCategory.OriginalData,
                                    reduction_values=f'[{item.embedding.strip('[]').replace(' ', ',')}]',
                                    has_enzyme=True,
                                    has_pathway=True,
                                    has_target=True,
                                    has_smiles=True) for item in embeddings]

        self.reduction_data_repository.insert_batch_check_duplicate(reduction_datas=reductions)
