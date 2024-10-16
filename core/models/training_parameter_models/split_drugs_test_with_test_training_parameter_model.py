from dataclasses import dataclass

from core.models.training_parameter_base_model import TrainingParameterBaseModel
from core.repository_models.training_drug_data_dto import TrainingDrugDataDTO
from core.repository_models.training_drug_interaction_dto import TrainingDrugInteractionDTO


@dataclass
class SplitDrugsTestWithTestTrainingParameterModel(TrainingParameterBaseModel):
    drug_data: list[TrainingDrugDataDTO]
    interaction_data: list[TrainingDrugInteractionDTO]
