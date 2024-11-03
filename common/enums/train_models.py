from enum import Enum

from common.enums.scenarios import Scenarios


class TrainModel(Enum):
    JoinBeforeSoftmax = (2, Scenarios.SplitInteractionSimilarities, 'Having n separate network, but before softmax layer, join the last layer.')
    SumSoftmaxOutputs = (3, Scenarios.SplitInteractionSimilarities, 'Have n separate network, and finally sum them.')
    AE_Con_DNN = (4, Scenarios.SplitInteractionSimilarities, 'Reduce data dimension, join them and send to DNN')
    Contact_DNN = (5, Scenarios.SplitInteractionSimilarities, 'Just join all data and send to DNN.')
    KNN = (6, Scenarios.SplitInteractionSimilarities, 'This network learns by KNN.')
    KNNWithAutoEncoder = (7, Scenarios.SplitInteractionSimilarities, 'This network learns by KNN and reduce dimension by AutoEncoder.')
    SVM = (8, Scenarios.SplitInteractionSimilarities, 'This network learns by SVM.')
    Con_AE_DNN = (9, Scenarios.SplitInteractionSimilarities, 'This network learns by Concat input data AutoEncoder and then DNN.')
    LR = (10, Scenarios.SplitInteractionSimilarities, 'Logistic Regression.')
    RF = (11, Scenarios.SplitInteractionSimilarities, 'Random Forest.')
    GAT_Enc_Con_DNN = (12, Scenarios.SplitInteractionSimilarities, 'GAT On SMILES, Encoder, Concat AutoEncoders and then DNN')
    GAT_MHA_DNN = (13, Scenarios.SplitInteractionSimilarities, 'GAT On SMILES, Multi-Head-Attention between text fields and other fields (Pathway, Target, '
                                                               'Enzyme) and then DNN.')
    GAT_AE_DNN = (14, Scenarios.SplitInteractionSimilarities, 'GAT On SMILES, AutoEncoder, Concat AutoEncoders and then DNN')
    MHA = (20, Scenarios.SplitInteractionSimilarities, 'Multi-Head Attention.')

    Drug_AE_Con_DNN = (204, Scenarios.SplitDrugsTestWithTrain, 'Reduce data dimension, join them and send to DNN')
    Drug_GAT_Enc_Con_DNN = (205, Scenarios.SplitDrugsTestWithTrain, 'GAT On SMILES, AutoEncoder, Concat AutoEncoders and then DNN')
    Drug_Deep_DDI = (210, Scenarios.SplitDrugsTestWithTrain, 'The Old Base Algorithm, just use SMILES code.')
    Drug_DDIMDL = (211, Scenarios.SplitDrugsTestWithTrain, 'The Old Base Algorithm, just use SMILES code.')

    Deep_DDI = (501, Scenarios.SplitInteractionSimilarities, 'The Old Base Algorithm, just use SMILES code.')
    DDIMDL = (502, Scenarios.SplitInteractionSimilarities, 'The Old Base Algorithm, just use SMILES code.')

    Test = (1000, Scenarios.SplitDrugsTestWithTrain, "This network is for test new algorithms.")

    def __init__(self, value, scenario: Scenarios, description):
        self._value_ = value
        self.scenario = scenario
        self.description = description

    @classmethod
    def from_value(cls, value):
        for item in cls:
            if item.value == value:
                return item
        raise ValueError(f"{value} is not a valid {cls.__name__}")
