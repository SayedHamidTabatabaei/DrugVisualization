from dataclasses import dataclass, field

from core.repository_models.training_drug_train_values_dto import TrainingDrugTrainValuesDTO


@dataclass
class TrainingDrugDataDTO:
    drug_id: int
    drugbank_id: int
    drug_name: str
    train_values: list[TrainingDrugTrainValuesDTO] = field(default_factory=list)
