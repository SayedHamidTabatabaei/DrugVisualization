from dataclasses import dataclass

from common.enums.loss_functions import LossFunctions
from common.enums.train_models import TrainModel


@dataclass
class TrainingResultDTO:
    id: int
    name: str
    description: str
    train_model: TrainModel
    loss_function: LossFunctions
    is_test_algorithm: bool
    class_weight: bool
    training_conditions: str
    model_parameters: str
    accuracy: float
    loss: float
    f1_score_weighted: float
    f1_score_micro: float
    f1_score_macro: float
    auc_weighted: float
    auc_micro: float
    auc_macro: float
    aupr_weighted: float
    aupr_micro: float
    aupr_macro: float
    recall_weighted: float
    recall_micro: float
    recall_macro: float
    precision_weighted: float
    precision_micro: float
    precision_macro: float
    execute_time: str
    min_sample_count: int
