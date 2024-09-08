from injector import inject
from tqdm import tqdm

from businesses import reduction_business
from businesses.base_business import BaseBusiness
from businesses.trains import train_plan1
from common.enums.category import Category
from common.enums.reduction_category import ReductionCategory
from common.enums.similarity_type import SimilarityType
from common.enums.text_type import TextType
from common.enums.train_models import TrainModel
from core.models.train_request_model import TrainRequestModel
from core.repository_models.interaction_dto import InteractionDTO
from core.repository_models.reduction_data_dto import ReductionDataDTO
from core.repository_models.training_data_dto import TrainingDataDTO
from infrastructure.repositories.reduction_data_repository import ReductionDataRepository


def map_training_data(interactions: list[InteractionDTO], reductions: list[ReductionDataDTO]) -> list[TrainingDataDTO]:
    reduction_dict = {reduction.drug_id: [float(val) for val in reduction.reduction_value[1:-1].split(',')]
                      for reduction in reductions}

    training_data = []

    for interaction in interactions:

        training_entity = TrainingDataDTO(drug_1=interaction.drug_1,
                                          drugbank_id_1=interaction.drugbank_id_1,
                                          reduction_values_1=reduction_dict.get(interaction.drug_1, None),
                                          drug_2=interaction.drug_2,
                                          drugbank_id_2=interaction.drugbank_id_2,
                                          reduction_values_2=reduction_dict.get(interaction.drug_2, None),
                                          interaction_type=interaction.interaction_type)
        training_data.append(training_entity)

    return training_data


class TrainingBusiness(BaseBusiness):
    @inject
    def __init__(self, reduction_repository: ReductionDataRepository):
        BaseBusiness.__init__(self)
        self.reduction_repository = reduction_repository

    def train(self, train_request: TrainRequestModel):
        match train_request.train_model:
            case TrainModel.SimpleOneInput:
                data = self.prepare_data(train_request)[0]
                train_plan1.train(data)
            case _:
                raise ValueError(f'Unsupported train model: {train_request.train_model}')

    def get_training_data(self, similarity_type: SimilarityType, category: Category,
                          reduction_category: ReductionCategory) -> list[TrainingDataDTO]:

        interactions = self.reduction_repository.find_interactions(similarity_type, category, reduction_category,
                                                                   True, True, True, True)

        reductions = self.reduction_repository.find_reductions(reduction_category, similarity_type, category,
                                                               True, True, True, True,
                                                               0, 100000)

        results = map_training_data(interactions, reductions)

        if not results:
            raise Exception(f"No training data found for {similarity_type} {category} {reduction_category}")

        return results

    def prepare_data(self, train_request):

        data = []

        if train_request.substructure_similarity and train_request.substructure_reduction:
            data.append(self.get_training_data(train_request.substructure_similarity, Category.Substructure,
                                               train_request.substructure_reduction))

        if train_request.target_similarity and train_request.target_reduction:
            data.append(self.get_training_data(train_request.target_similarity, Category.Target,
                                               train_request.target_reduction))

        if train_request.enzyme_similarity and train_request.enzyme_reduction:
            data.append(self.get_training_data(train_request.enzyme_similarity, Category.Enzyme,
                                               train_request.enzyme_reduction))

        if train_request.pathway_similarity and train_request.pathway_reduction:
            data.append(self.get_training_data(train_request.pathway_similarity, Category.Pathway,
                                               train_request.pathway_reduction))

        if train_request.description_embedding and train_request.description_reduction:
            category = reduction_business.find_category(train_request.description_embedding, TextType.Description)
            data.append(self.get_training_data(SimilarityType.Original, category, train_request.description_reduction))

        if train_request.indication_embedding and train_request.indication_reduction:
            category = reduction_business.find_category(train_request.indication_embedding, TextType.Indication)
            data.append(self.get_training_data(SimilarityType.Original, category, train_request.indication_reduction))

        if train_request.pharmacodynamics_embedding and train_request.pharmacodynamics_reduction:
            category = reduction_business.find_category(train_request.pharmacodynamics_embedding,
                                                        TextType.Pharmacodynamics)
            data.append(self.get_training_data(SimilarityType.Original, category,
                                               train_request.pharmacodynamics_reduction))

        if train_request.mechanism_of_action_embedding and train_request.mechanism_of_action_reduction:
            category = reduction_business.find_category(train_request.mechanism_of_action_embedding,
                                                        TextType.MechanismOfAction)
            data.append(self.get_training_data(SimilarityType.Original, category,
                                               train_request.mechanism_of_action_reduction))

        if train_request.toxicity_embedding and train_request.toxicity_reduction:
            category = reduction_business.find_category(train_request.toxicity_embedding, TextType.Toxicity)
            data.append(self.get_training_data(SimilarityType.Original, category, train_request.toxicity_reduction))

        if train_request.metabolism_embedding and train_request.metabolism_reduction:
            category = reduction_business.find_category(train_request.metabolism_embedding, TextType.Metabolism)
            data.append(self.get_training_data(SimilarityType.Original, category, train_request.metabolism_reduction))

        if train_request.absorption_embedding and train_request.absorption_reduction:
            category = reduction_business.find_category(train_request.absorption_embedding, TextType.Absorption)
            data.append(self.get_training_data(SimilarityType.Original, category, train_request.absorption_reduction))

        if train_request.half_life_embedding and train_request.half_life_reduction:
            category = reduction_business.find_category(train_request.half_life_embedding, TextType.HalfLife)
            data.append(self.get_training_data(SimilarityType.Original, category, train_request.half_life_reduction))

        if train_request.protein_binding_embedding and train_request.protein_binding_reduction:
            category = reduction_business.find_category(train_request.protein_binding_embedding,
                                                        TextType.ProteinBinding)
            data.append(self.get_training_data(SimilarityType.Original, category,
                                               train_request.protein_binding_reduction))

        if train_request.route_of_elimination_embedding and train_request.route_of_elimination_reduction:
            category = reduction_business.find_category(train_request.route_of_elimination_embedding,
                                                        TextType.RouteOfElimination)
            data.append(self.get_training_data(SimilarityType.Original, category,
                                               train_request.route_of_elimination_reduction))

        if train_request.volume_of_distribution_embedding and train_request.volume_of_distribution_reduction:
            category = reduction_business.find_category(train_request.volume_of_distribution_embedding,
                                                        TextType.VolumeOfDistribution)
            data.append(self.get_training_data(SimilarityType.Original, category,
                                               train_request.volume_of_distribution_reduction))

        if train_request.clearance_embedding and train_request.clearance_reduction:
            category = reduction_business.find_category(train_request.clearance_embedding, TextType.Clearance)
            data.append(self.get_training_data(SimilarityType.Original, category, train_request.clearance_reduction))

        if train_request.classification_description_embedding and train_request.classification_description_reduction:
            category = reduction_business.find_category(train_request.classification_description_embedding,
                                                        TextType.ClassificationDescription)
            data.append(self.get_training_data(SimilarityType.Original, category,
                                               train_request.classification_description_reduction))

        return data
