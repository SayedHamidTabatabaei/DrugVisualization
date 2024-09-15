from enum import Enum


class TrainModel(Enum):
    SimpleOneInput = 1
    JoinSimplesBeforeSoftmax = 2
    SumSoftmaxOutputs = 3
    AutoEncoderWithDNN = 4
    ContactDataWithOneDNN = 5
