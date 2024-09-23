from enum import Enum


class TrainModel(Enum):
    SimpleOneInput = (1, 'Just learn one input by DNN.')
    JoinSimplesBeforeSoftmax = (2, 'Having n separate network, but before softmax layer, join the last layer.')
    SumSoftmaxOutputs = (3, 'Have n separate network, and finally sum them.')
    AutoEncoderWithDNN = (4, 'Reduce data dimension, join them and send to DNN')
    ContactDataWithOneDNN = (5, 'Just join all data and send to DNN.')
    KNN = (6, 'This network learns by KNN.')
    KNNWithAutoEncoder = (7, 'This network learns by KNN and reduce dimension by AutoEncoder.')
    SVM = (8, "This network learns by SVM.")
    

    def __init__(self, value, description):
        self._value_ = value
        self.description = description

    @classmethod
    def from_value(cls, value):
        for item in cls:
            if item.value == value:
                return item
        raise ValueError(f"{value} is not a valid {cls.__name__}")
