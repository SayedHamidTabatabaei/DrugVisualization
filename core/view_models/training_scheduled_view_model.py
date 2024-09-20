from dataclasses import dataclass


@dataclass
class TrainingScheduledViewModel:
    id: int
    name: str
    description: str
    train_model: str
    training_conditions: str
    schedule_date: str
