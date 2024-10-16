from injector import inject
from tqdm import tqdm

from businesses.base_business import BaseBusiness
from businesses.embedding_services import embedding_instances
from common.enums.embedding_type import EmbeddingType
from common.enums.text_type import TextType
from core.domain.drug_embedding import DrugEmbedding
from core.domain.interaction_embedding import InteractionEmbedding
from infrastructure.repositories.drug_embedding_repository import DrugEmbeddingRepository
from infrastructure.repositories.drug_interaction_repository import DrugInteractionRepository
from infrastructure.repositories.drug_repository import DrugRepository
from infrastructure.repositories.interaction_embedding_repository import InteractionEmbeddingRepository
from infrastructure.repositories.reduction_data_repository import ReductionDataRepository


class DrugEmbeddingBusiness(BaseBusiness):
    @inject
    def __init__(self, drug_embedding_repository: DrugEmbeddingRepository, drug_repository: DrugRepository,
                 interaction_embedding_repository: InteractionEmbeddingRepository, drug_interaction_repository: DrugInteractionRepository,
                 final_data_repository: ReductionDataRepository):
        BaseBusiness.__init__(self)
        self.final_data_repository = final_data_repository
        self.drug_embedding_repository = drug_embedding_repository
        self.drug_repository = drug_repository
        self.interaction_embedding_repository = interaction_embedding_repository
        self.drug_interaction_repository = drug_interaction_repository

    def get_all_embeddings(self, embedding_type: EmbeddingType, text_type: TextType, start: int, length: int):

        if text_type != TextType.InteractionDescription:
            total_number = self.drug_embedding_repository.get_embedding_count(embedding_type, text_type)

            embeddings = self.get_embedding_by_text_type(embedding_type, text_type, start, length)
        else:
            total_number = self.interaction_embedding_repository.get_embedding_count(embedding_type, text_type)

            embeddings = self.get_embedding_by_text_type(embedding_type, text_type, start, length)

        return embeddings, total_number

    def get_embedding_by_text_type(self, embedding_type: EmbeddingType, text_type: TextType, start: int, length: int):
        if text_type == TextType.Description:
            return self.drug_embedding_repository.find_drug_embedding_description(embedding_type, start, length)
        elif text_type == TextType.Indication:
            return self.drug_embedding_repository.find_drug_embedding_indication(embedding_type, start, length)
        elif text_type == TextType.Pharmacodynamics:
            return self.drug_embedding_repository.find_drug_embedding_pharmacodynamics(embedding_type, start, length)
        elif text_type == TextType.MechanismOfAction:
            return self.drug_embedding_repository.find_drug_embedding_mechanism_of_action(embedding_type, start, length)
        elif text_type == TextType.Toxicity:
            return self.drug_embedding_repository.find_drug_embedding_toxicity(embedding_type, start, length)
        elif text_type == TextType.Metabolism:
            return self.drug_embedding_repository.find_drug_embedding_metabolism(embedding_type, start, length)
        elif text_type == TextType.Absorption:
            return self.drug_embedding_repository.find_drug_embedding_absorption(embedding_type, start, length)
        elif text_type == TextType.HalfLife:
            return self.drug_embedding_repository.find_drug_embedding_half_life(embedding_type, start, length)
        elif text_type == TextType.ProteinBinding:
            return self.drug_embedding_repository.find_drug_embedding_protein_binding(embedding_type, start, length)
        elif text_type == TextType.RouteOfElimination:
            return self.drug_embedding_repository.find_drug_embedding_route_of_elimination(embedding_type, start, length)
        elif text_type == TextType.VolumeOfDistribution:
            return self.drug_embedding_repository.find_drug_embedding_volume_of_distribution(embedding_type, start, length)
        elif text_type == TextType.Clearance:
            return self.drug_embedding_repository.find_drug_embedding_clearance(embedding_type, start, length)
        elif text_type == TextType.ClassificationDescription:
            return self.drug_embedding_repository.find_drug_embedding_classification_description(embedding_type, start, length)
        elif text_type == TextType.InteractionDescription:
            return self.interaction_embedding_repository.find_interaction_embedding_description(embedding_type, start, length)

    def calculate_embeddings(self, embedding_type: EmbeddingType, text_type: TextType):
        match text_type:
            case TextType.Description:
                drugs = self.drug_repository.find_all_drug_description()

            case TextType.Indication:
                drugs = self.drug_repository.find_all_drug_indication()

            case TextType.Pharmacodynamics:
                drugs = self.drug_repository.find_all_drug_pharmacodynamics()

            case TextType.MechanismOfAction:
                drugs = self.drug_repository.find_all_drug_mechanism_of_action()

            case TextType.Toxicity:
                drugs = self.drug_repository.find_all_drug_toxicity()

            case TextType.Metabolism:
                drugs = self.drug_repository.find_all_drug_metabolism()

            case TextType.Absorption:
                drugs = self.drug_repository.find_all_drug_absorption()

            case TextType.HalfLife:
                drugs = self.drug_repository.find_all_drug_half_life()

            case TextType.ProteinBinding:
                drugs = self.drug_repository.find_all_drug_protein_binding()

            case TextType.RouteOfElimination:
                drugs = self.drug_repository.find_all_drug_route_of_elimination()

            case TextType.VolumeOfDistribution:
                drugs = self.drug_repository.find_all_drug_volume_of_distribution()

            case TextType.Clearance:
                drugs = self.drug_repository.find_all_drug_clearance()

            case TextType.ClassificationDescription:
                drugs = self.drug_repository.find_all_drug_classification_description()

            case TextType.InteractionDescription:
                interactions = self.drug_interaction_repository.find_all_interaction_description()
                self.generate_interaction_embedding(interactions, embedding_type, text_type)
                return

            case _:
                raise

        self.generate_embedding(drugs, embedding_type, text_type)

    def generate_embedding(self, drugs, embedding_type: EmbeddingType, text_type: TextType):

        instance = embedding_instances.get_instance(embedding_type)

        drug_embeddings: list[DrugEmbedding] = []
        for drug in tqdm(drugs, desc=f"Processing embedding ({text_type.name} - {embedding_type.name})"):
            embedding, issue_on_max_length = instance.embed(drug.text)

            drug_embeddings.append(DrugEmbedding(drug_id=drug.id,
                                                 embedding_type=embedding_type,
                                                 text_type=text_type,
                                                 embedding=embedding,
                                                 issue_on_max_length=issue_on_max_length))

        self.drug_embedding_repository.insert_batch_check_duplicate(
            drug_embeddings, [DrugEmbedding.embedding, DrugEmbedding.issue_on_max_length])

    def generate_interaction_embedding(self, interactions, embedding_type: EmbeddingType, text_type: TextType):

        instance = embedding_instances.get_instance(embedding_type)

        interaction_embeddings: list[InteractionEmbedding] = []
        for interaction in tqdm(interactions, desc=f"Processing embedding ({text_type.name} - {embedding_type.name})"):
            embedding, issue_on_max_length = instance.embed(interaction.text)

            interaction_embeddings.append(InteractionEmbedding(interaction_id=interaction.id,
                                                               embedding_type=embedding_type,
                                                               text_type=text_type,
                                                               embedding=embedding,
                                                               issue_on_max_length=issue_on_max_length))

        self.interaction_embedding_repository.insert_batch_check_duplicate(
            interaction_embeddings, [InteractionEmbedding.embedding, InteractionEmbedding.issue_on_max_length])
