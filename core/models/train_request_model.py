import json
from dataclasses import dataclass, field, asdict
from typing import Optional

from common.enums.embedding_type import EmbeddingType
from common.enums.reduction_category import ReductionCategory
from common.enums.similarity_type import SimilarityType
from common.enums.train_models import TrainModel


@dataclass
class TrainRequestModel:
    train_model: TrainModel
    substructure_similarity: Optional[SimilarityType] = field(default=None)
    substructure_reduction: Optional[ReductionCategory] = field(default=None)
    target_similarity: Optional[SimilarityType] = field(default=None)
    target_reduction: Optional[ReductionCategory] = field(default=None)
    enzyme_similarity: Optional[SimilarityType] = field(default=None)
    enzyme_reduction: Optional[ReductionCategory] = field(default=None)
    pathway_similarity: Optional[SimilarityType] = field(default=None)
    pathway_reduction: Optional[ReductionCategory] = field(default=None)
    description_embedding: Optional[EmbeddingType] = field(default=None)
    description_reduction: Optional[ReductionCategory] = field(default=None)
    indication_embedding: Optional[EmbeddingType] = field(default=None)
    indication_reduction: Optional[ReductionCategory] = field(default=None)
    pharmacodynamics_embedding: Optional[EmbeddingType] = field(default=None)
    pharmacodynamics_reduction: Optional[ReductionCategory] = field(default=None)
    mechanism_of_action_embedding: Optional[EmbeddingType] = field(default=None)
    mechanism_of_action_reduction: Optional[ReductionCategory] = field(default=None)
    toxicity_embedding: Optional[EmbeddingType] = field(default=None)
    toxicity_reduction: Optional[ReductionCategory] = field(default=None)
    metabolism_embedding: Optional[EmbeddingType] = field(default=None)
    metabolism_reduction: Optional[ReductionCategory] = field(default=None)
    absorption_embedding: Optional[EmbeddingType] = field(default=None)
    absorption_reduction: Optional[ReductionCategory] = field(default=None)
    half_life_embedding: Optional[EmbeddingType] = field(default=None)
    half_life_reduction: Optional[ReductionCategory] = field(default=None)
    protein_binding_embedding: Optional[EmbeddingType] = field(default=None)
    protein_binding_reduction: Optional[ReductionCategory] = field(default=None)
    route_of_elimination_embedding: Optional[EmbeddingType] = field(default=None)
    route_of_elimination_reduction: Optional[ReductionCategory] = field(default=None)
    volume_of_distribution_embedding: Optional[EmbeddingType] = field(default=None)
    volume_of_distribution_reduction: Optional[ReductionCategory] = field(default=None)
    clearance_embedding: Optional[EmbeddingType] = field(default=None)
    clearance_reduction: Optional[ReductionCategory] = field(default=None)
    classification_description_embedding: Optional[EmbeddingType] = field(default=None)
    classification_description_reduction: Optional[ReductionCategory] = field(default=None)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            train_model=TrainModel[data.get('train_model')],
            substructure_similarity=SimilarityType[data['substructure_similarity']] if 'substructure_similarity' in data else None,
            substructure_reduction=ReductionCategory[data['substructure_reduction']] if 'substructure_reduction' in data else None,
            target_similarity=SimilarityType[data['target_similarity']] if 'target_similarity' in data else None,
            target_reduction=ReductionCategory[data['target_reduction']] if 'target_reduction' in data else None,
            enzyme_similarity=SimilarityType[data['enzyme_similarity']] if 'enzyme_similarity' in data else None,
            enzyme_reduction=ReductionCategory[data['enzyme_reduction']] if 'enzyme_reduction' in data else None,
            pathway_similarity=SimilarityType[data['pathway_similarity']] if 'pathway_similarity' in data else None,
            pathway_reduction=ReductionCategory[data['pathway_reduction']] if 'pathway_reduction' in data else None,
            description_embedding=EmbeddingType[data['description_embedding']] if 'description_embedding' in data else None,
            description_reduction=ReductionCategory[data['description_reduction']] if 'description_reduction' in data else None,
            indication_embedding=EmbeddingType[data['indication_embedding']] if 'indication_embedding' in data else None,
            indication_reduction=ReductionCategory[data['indication_reduction']] if 'indication_reduction' in data else None,
            pharmacodynamics_embedding=EmbeddingType[data['pharmacodynamics_embedding']] if 'pharmacodynamics_embedding' in data else None,
            pharmacodynamics_reduction=ReductionCategory[data['pharmacodynamics_reduction']] if 'pharmacodynamics_reduction' in data else None,
            mechanism_of_action_embedding=EmbeddingType[data['mechanism_of_action_embedding']] if 'mechanism_of_action_embedding' in data else None,
            mechanism_of_action_reduction=ReductionCategory[data['mechanism_of_action_reduction']] if 'mechanism_of_action_reduction' in data else None,
            toxicity_embedding=EmbeddingType[data['toxicity_embedding']] if 'toxicity_embedding' in data else None,
            toxicity_reduction=ReductionCategory[data['toxicity_reduction']] if 'toxicity_reduction' in data else None,
            metabolism_embedding=EmbeddingType[data['metabolism_embedding']] if 'metabolism_embedding' in data else None,
            metabolism_reduction=ReductionCategory[data['metabolism_reduction']] if 'metabolism_reduction' in data else None,
            absorption_embedding=EmbeddingType[data['absorption_embedding']] if 'absorption_embedding' in data else None,
            absorption_reduction=ReductionCategory[data['absorption_reduction']] if 'absorption_reduction' in data else None,
            half_life_embedding=EmbeddingType[data['half_life_embedding']] if 'half_life_embedding' in data else None,
            half_life_reduction=ReductionCategory[data['half_life_reduction']] if 'half_life_reduction' in data else None,
            protein_binding_embedding=EmbeddingType[data['protein_binding_embedding']] if 'protein_binding_embedding' in data else None,
            protein_binding_reduction=ReductionCategory[data['protein_binding_reduction']] if 'protein_binding_reduction' in data else None,
            route_of_elimination_embedding=EmbeddingType[data['route_of_elimination_embedding']] if 'route_of_elimination_embedding' in data else None,
            route_of_elimination_reduction=ReductionCategory[data['route_of_elimination_reduction']] if 'route_of_elimination_reduction' in data else None,
            volume_of_distribution_embedding=EmbeddingType[data['volume_of_distribution_embedding']] if 'volume_of_distribution_embedding' in data else None,
            volume_of_distribution_reduction=ReductionCategory[data['volume_of_distribution_reduction']] if 'volume_of_distribution_reduction' in data else None,
            clearance_embedding=EmbeddingType[data['clearance_embedding']] if 'clearance_embedding' in data else None,
            clearance_reduction=ReductionCategory[data['clearance_reduction']] if 'clearance_reduction' in data else None,
            classification_description_embedding=EmbeddingType[data['classification_description_embedding']] if 'classification_description_embedding' in data else None,
            classification_description_reduction=ReductionCategory[data['classification_description_reduction']] if 'classification_description_reduction' in data else None
            )

    def to_json(self):
        return json.dumps(asdict(self), default=str, indent=4)
