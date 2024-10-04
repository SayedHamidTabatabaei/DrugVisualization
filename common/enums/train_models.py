from enum import Enum


class TrainModel(Enum):
    SimpleOneInput = (1, 'Just learn one input by DNN.')
    JoinBeforeSoftmax = (2, 'Having n separate network, but before softmax layer, join the last layer.')
    SumSoftmaxOutputs = (3, 'Have n separate network, and finally sum them.')
    AE_Con_DNN = (4, 'Reduce data dimension, join them and send to DNN')
    Contact_DNN = (5, 'Just join all data and send to DNN.')
    KNN = (6, 'This network learns by KNN.')
    KNNWithAutoEncoder = (7, 'This network learns by KNN and reduce dimension by AutoEncoder.')
    SVM = (8, "This network learns by SVM.")
    Con_AE_DNN = (9, 'This network learns by Concat input data AutoEncoder and then DNN.')
    LR = (10, "Logistic Regression.")
    RF = (11, "Random Forest.")

    Test = (1000, "This network is for test new algorithms.")

    def __init__(self, value, description):
        self._value_ = value
        self.description = description

    @classmethod
    def from_value(cls, value):
        for item in cls:
            if item.value == value:
                return item
        raise ValueError(f"{value} is not a valid {cls.__name__}")
