import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

from common.enums.embedding_type import EmbeddingType
from common.enums.loss_functions import LossFunctions
from common.enums.similarity_type import SimilarityType
from common.enums.train_models import TrainModel


@dataclass
class TrainRequestViewModel:
    train_model: TrainModel
    loss_function: LossFunctions
    class_weight: bool
    name: str
    description: str
    is_test_algorithm: bool
    substructure_similarity: Optional[SimilarityType] = field(default=None)
    target_similarity: Optional[SimilarityType] = field(default=None)
    enzyme_similarity: Optional[SimilarityType] = field(default=None)
    pathway_similarity: Optional[SimilarityType] = field(default=None)
    description_embedding: Optional[EmbeddingType] = field(default=None)
    indication_embedding: Optional[EmbeddingType] = field(default=None)
    pharmacodynamics_embedding: Optional[EmbeddingType] = field(default=None)
    mechanism_of_action_embedding: Optional[EmbeddingType] = field(default=None)
    toxicity_embedding: Optional[EmbeddingType] = field(default=None)
    metabolism_embedding: Optional[EmbeddingType] = field(default=None)
    absorption_embedding: Optional[EmbeddingType] = field(default=None)
    half_life_embedding: Optional[EmbeddingType] = field(default=None)
    protein_binding_embedding: Optional[EmbeddingType] = field(default=None)
    route_of_elimination_embedding: Optional[EmbeddingType] = field(default=None)
    volume_of_distribution_embedding: Optional[EmbeddingType] = field(default=None)
    clearance_embedding: Optional[EmbeddingType] = field(default=None)
    classification_description_embedding: Optional[EmbeddingType] = field(default=None)

    @classmethod
    def from_dict(cls, data: dict):

        loss_function = None
        if data.get('loss_function'):
            if str(data.get('loss_function')).isdigit():
                loss_function = LossFunctions.from_value(int(data.get('loss_function')))
            else:
                loss_function = LossFunctions[data.get('loss_function')]

        return cls(
            train_model=TrainModel[data.get('train_model')],
            loss_function=loss_function,
            name=data.get('model_name'),
            description=data.get('model_description'),
            class_weight=data.get('class_weight'),
            is_test_algorithm=data.get('is_test_algorithm'),
            substructure_similarity=SimilarityType[data['substructure_similarity']] if 'substructure_similarity' in data and data['substructure_similarity'] else None,
            target_similarity=SimilarityType[data['target_similarity']] if 'target_similarity' in data and data['target_similarity'] else None,
            enzyme_similarity=SimilarityType[data['enzyme_similarity']] if 'enzyme_similarity' in data and data['enzyme_similarity'] else None,
            pathway_similarity=SimilarityType[data['pathway_similarity']] if 'pathway_similarity' in data and data['pathway_similarity'] else None,
            description_embedding=EmbeddingType[data['description_embedding']] if 'description_embedding' in data and data['description_embedding'] else None,
            indication_embedding=EmbeddingType[data['indication_embedding']] if 'indication_embedding' in data and data['indication_embedding'] else None,
            pharmacodynamics_embedding=EmbeddingType[data['pharmacodynamics_embedding']] if 'pharmacodynamics_embedding' in data and data['pharmacodynamics_embedding'] else None,
            mechanism_of_action_embedding=EmbeddingType[data['mechanism_of_action_embedding']] if 'mechanism_of_action_embedding' in data and data['mechanism_of_action_embedding'] else None,
            toxicity_embedding=EmbeddingType[data['toxicity_embedding']] if 'toxicity_embedding' in data and data['toxicity_embedding'] else None,
            metabolism_embedding=EmbeddingType[data['metabolism_embedding']] if 'metabolism_embedding' in data and data['metabolism_embedding'] else None,
            absorption_embedding=EmbeddingType[data['absorption_embedding']] if 'absorption_embedding' in data and data['absorption_embedding'] else None,
            half_life_embedding=EmbeddingType[data['half_life_embedding']] if 'half_life_embedding' in data and data['half_life_embedding'] else None,
            protein_binding_embedding=EmbeddingType[data['protein_binding_embedding']] if 'protein_binding_embedding' in data and data['protein_binding_embedding'] else None,
            route_of_elimination_embedding=EmbeddingType[data['route_of_elimination_embedding']] if 'route_of_elimination_embedding' in data and data['route_of_elimination_embedding'] else None,
            volume_of_distribution_embedding=EmbeddingType[data['volume_of_distribution_embedding']] if 'volume_of_distribution_embedding' in data and data['volume_of_distribution_embedding'] else None,
            clearance_embedding=EmbeddingType[data['clearance_embedding']] if 'clearance_embedding' in data and data['clearance_embedding'] else None,
            classification_description_embedding=EmbeddingType[data['classification_description_embedding']] if 'classification_description_embedding' in data and data['classification_description_embedding'] else None,
            )

    def to_json(self):
        return json.dumps(asdict(self),
                          default=lambda o: o.name if isinstance(o, Enum) else str(o),
                          indent=4)

    @classmethod
    def from_json(cls, json_str: str):
        data = json.loads(json_str)
        return cls.from_dict(data)
