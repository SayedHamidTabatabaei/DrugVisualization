from injector import inject
from tqdm import tqdm

from businesses.base_business import BaseBusiness
from businesses.embedding_services import embedding_instances
from common.enums.embedding_type import EmbeddingType
from common.enums.text_type import TextType
from core.domain.drug_embedding import DrugEmbedding
from infrastructure.repositories.drug_embedding_repository import DrugEmbeddingRepository
from infrastructure.repositories.drug_repository import DrugRepository
from infrastructure.repositories.reduction_data_repository import ReductionDataRepository


class DrugEmbeddingBusiness(BaseBusiness):
    @inject
    def __init__(self, drug_embedding_repository: DrugEmbeddingRepository, drug_repository: DrugRepository,
                 final_data_repository: ReductionDataRepository):
        BaseBusiness.__init__(self)
        self.final_data_repository = final_data_repository
        self.drug_embedding_repository = drug_embedding_repository
        self.drug_repository = drug_repository

    def get_all_embeddings(self, embedding_type: EmbeddingType, text_type: TextType, start: int, length: int):
        total_number = self.drug_embedding_repository.get_embedding_count(embedding_type, text_type)

        embeddings = self.get_embedding_by_text_type(embedding_type, text_type, start, length)

        return embeddings, total_number

    def get_embedding_by_text_type(self, embedding_type: EmbeddingType, text_type: TextType, start: int, length: int):
        if text_type == TextType.Description:
            return self.drug_embedding_repository.find_drug_embedding_description(embedding_type, start, length)
        elif text_type == TextType.Indication:
            return self.drug_embedding_repository.find_drug_embedding_indication(embedding_type, start, length)
        elif text_type == TextType.Pharmacodynamics:
            return self.drug_embedding_repository.find_drug_embedding_pharmacodynamics(embedding_type, start,
                                                                                       length)
        elif text_type == TextType.MechanismOfAction:
            return self.drug_embedding_repository.find_drug_embedding_mechanism_of_action(embedding_type, start,
                                                                                          length)
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
            return self.drug_embedding_repository.find_drug_embedding_route_of_elimination(embedding_type, start,
                                                                                           length)
        elif text_type == TextType.VolumeOfDistribution:
            return self.drug_embedding_repository.find_drug_embedding_volume_of_distribution(embedding_type, start,
                                                                                             length)
        elif text_type == TextType.Clearance:
            return self.drug_embedding_repository.find_drug_embedding_clearance(embedding_type, start, length)
        elif text_type == TextType.ClassificationDescription:
            return self.drug_embedding_repository.find_drug_embedding_classification_description(embedding_type,
                                                                                                 start, length)

    def calculate_embeddings(self, embedding_type: EmbeddingType, text_type: TextType):
        match (embedding_type, text_type):
            case (EmbeddingType.PubMedBERT, TextType.Description):
                drugs = self.drug_repository.find_all_drug_description()

            case (EmbeddingType.PubMedBERT, TextType.Indication):
                drugs = self.drug_repository.find_all_drug_indication()

            case (EmbeddingType.PubMedBERT, TextType.Pharmacodynamics):
                drugs = self.drug_repository.find_all_drug_pharmacodynamics()

            case (EmbeddingType.PubMedBERT, TextType.MechanismOfAction):
                drugs = self.drug_repository.find_all_drug_mechanism_of_action()

            case (EmbeddingType.PubMedBERT, TextType.Toxicity):
                drugs = self.drug_repository.find_all_drug_toxicity()

            case (EmbeddingType.PubMedBERT, TextType.Metabolism):
                drugs = self.drug_repository.find_all_drug_metabolism()

            case (EmbeddingType.PubMedBERT, TextType.Absorption):
                drugs = self.drug_repository.find_all_drug_absorption()

            case (EmbeddingType.PubMedBERT, TextType.HalfLife):
                drugs = self.drug_repository.find_all_drug_half_life()

            case (EmbeddingType.PubMedBERT, TextType.ProteinBinding):
                drugs = self.drug_repository.find_all_drug_protein_binding()

            case (EmbeddingType.PubMedBERT, TextType.RouteOfElimination):
                drugs = self.drug_repository.find_all_drug_route_of_elimination()

            case (EmbeddingType.PubMedBERT, TextType.VolumeOfDistribution):
                drugs = self.drug_repository.find_all_drug_volume_of_distribution()

            case (EmbeddingType.PubMedBERT, TextType.Clearance):
                drugs = self.drug_repository.find_all_drug_clearance()

            case (EmbeddingType.PubMedBERT, TextType.ClassificationDescription):
                drugs = self.drug_repository.find_all_drug_classification_description()

            case (EmbeddingType.SciBERT, TextType.Description):
                drugs = self.drug_repository.find_all_drug_description()

            case (EmbeddingType.SciBERT, TextType.Indication):
                drugs = self.drug_repository.find_all_drug_indication()

            case (EmbeddingType.SciBERT, TextType.Pharmacodynamics):
                drugs = self.drug_repository.find_all_drug_pharmacodynamics()

            case (EmbeddingType.SciBERT, TextType.MechanismOfAction):
                drugs = self.drug_repository.find_all_drug_mechanism_of_action()

            case (EmbeddingType.SciBERT, TextType.Toxicity):
                drugs = self.drug_repository.find_all_drug_toxicity()

            case (EmbeddingType.SciBERT, TextType.Metabolism):
                drugs = self.drug_repository.find_all_drug_metabolism()

            case (EmbeddingType.SciBERT, TextType.Absorption):
                drugs = self.drug_repository.find_all_drug_absorption()

            case (EmbeddingType.SciBERT, TextType.HalfLife):
                drugs = self.drug_repository.find_all_drug_half_life()

            case (EmbeddingType.SciBERT, TextType.ProteinBinding):
                drugs = self.drug_repository.find_all_drug_protein_binding()

            case (EmbeddingType.SciBERT, TextType.RouteOfElimination):
                drugs = self.drug_repository.find_all_drug_route_of_elimination()

            case (EmbeddingType.SciBERT, TextType.VolumeOfDistribution):
                drugs = self.drug_repository.find_all_drug_volume_of_distribution()

            case (EmbeddingType.SciBERT, TextType.Clearance):
                drugs = self.drug_repository.find_all_drug_clearance()

            case (EmbeddingType.SciBERT, TextType.ClassificationDescription):
                drugs = self.drug_repository.find_all_drug_classification_description()

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
