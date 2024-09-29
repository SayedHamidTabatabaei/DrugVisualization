from dataclasses import dataclass


@dataclass
class TrainingParameterModel:
    train_id: int
    is_test_algorithm: bool
