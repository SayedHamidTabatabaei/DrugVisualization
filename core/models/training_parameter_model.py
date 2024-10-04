from dataclasses import dataclass

from common.enums.loss_functions import LossFunctions


@dataclass
class TrainingParameterModel:
    train_id: int
    loss_function: LossFunctions
    class_weight: bool
    is_test_algorithm: bool
