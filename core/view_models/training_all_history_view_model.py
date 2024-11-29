from dataclasses import dataclass


@dataclass
class TrainingAllHistoryViewModel:
    id: int
    name: str
    description: str
    train_model: str
    loss_function: str
    class_weight: bool
    execute_time: str
    min_sample_count: int
    training_results_count: int
    training_result_details_count: int
    is_completed: bool = False

    def __post_init__(self):
        self.is_completed = self.training_results_count > 0 and self.training_result_details_count > 0
