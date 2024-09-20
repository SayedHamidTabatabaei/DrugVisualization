from enum import Enum


class ReductionCategory(Enum):
    OriginalData = 1
    AutoEncoder_Max = 2
    AutoEncoder_Min = 3
    AutoEncoder_Mean = 4
