from enum import Enum


class Scenarios(Enum):
    SplitInteractionSimilarities = (1, "In this scenario splits by interactions and use similarity algorithms and splits 80-20.")
    SplitDrugsTestWithTrain = (2, "In this scenario splits by drugs and use similarity algorithms and test 'test set' with 'train set' and splits by k-fold.")
    SplitDrugsTestWithTest = (3, "In this scenario splits by drugs and use similarity algorithms and test 'test set' with 'test set' and splits by k-fold.")

    def __init__(self, value, description):
        self._value_ = value
        self.description = description

    @classmethod
    def from_value(cls, value):
        for item in cls:
            if item.value == value:
                return item
        raise ValueError(f"{value} is not a valid {cls.__name__}")
