from enum import Enum

from common.enums.scenarios import Scenarios


class TrainModel(Enum):
    JoinBeforeSoftmax = (2, Scenarios.SplitInteractionSimilarities, 'Having n separate network, but before softmax layer, join the last layer.')
    SumSoftmaxOutputs = (3, Scenarios.SplitInteractionSimilarities, 'Have n separate network, and finally sum them.')
    Enc_Con_DNN = (4, Scenarios.SplitInteractionSimilarities, 'Reduce data dimension, join them and send to DNN')
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

    Drug_Enc_Con_DNN = (204, Scenarios.SplitDrugsTestWithTrain, 'Reduce data dimension, join them and send to DNN')
    Drug_GAT_Enc_Con_DNN = (205, Scenarios.SplitDrugsTestWithTrain, 'GAT On SMILES, AutoEncoder, Concat AutoEncoders and then DNN')
    Drug_Deep_DDI = (210, Scenarios.SplitDrugsTestWithTrain, 'The Old Base Algorithm, just use SMILES code.')
    Drug_DDIMDL = (211, Scenarios.SplitDrugsTestWithTrain, 'The Old Base Algorithm, just use SMILES code.')
    Drug_CNN_DDI = (212, Scenarios.SplitDrugsTestWithTrain, 'The Old Base Algorithm, use CNN.')
    Drug_KNN = (251, Scenarios.SplitDrugsTestWithTrain, 'This network learns by KNN.')
    Drug_SVM = (252, Scenarios.SplitDrugsTestWithTrain, 'This network learns by SVM.')
    Drug_LR = (253, Scenarios.SplitDrugsTestWithTrain, 'Logistic Regression.')
    Drug_RF = (254, Scenarios.SplitDrugsTestWithTrain, 'Random Forest.')

    Drug_Enc_Con_DNN_Test = (304, Scenarios.SplitDrugsTestWithTest, 'Reduce data dimension, join them and send to DNN')
    Drug_GAT_Enc_Con_DNN_Test = (305, Scenarios.SplitDrugsTestWithTest, 'GAT On SMILES, AutoEncoder, Concat AutoEncoders and then DNN')
    Drug_Deep_DDI_Test = (310, Scenarios.SplitDrugsTestWithTest, 'The Old Base Algorithm, just use SMILES code.')
    Drug_DDIMDL_Test = (311, Scenarios.SplitDrugsTestWithTest, 'The Old Base Algorithm, just use SMILES code.')
    Drug_CNN_DDI_Test = (312, Scenarios.SplitDrugsTestWithTest, 'The Old Base Algorithm, use CNN.')
    Drug_KNN_Test = (351, Scenarios.SplitDrugsTestWithTest, 'This network learns by KNN.')
    Drug_SVM_Test = (352, Scenarios.SplitDrugsTestWithTest, 'This network learns by SVM.')
    Drug_LR_Test = (353, Scenarios.SplitDrugsTestWithTest, 'Logistic Regression.')
    Drug_RF_Test = (354, Scenarios.SplitDrugsTestWithTest, 'Random Forest.')

    Fold_GAT_Enc_Con_DNN = (405, Scenarios.FoldInteractionSimilarities, 'GAT On SMILES, AutoEncoder, Concat AutoEncoders and then DNN')
    Fold_Deep_DDI = (410, Scenarios.FoldInteractionSimilarities, 'The Old Base Algorithm, just use SMILES code.')
    Fold_DDIMDL = (411, Scenarios.FoldInteractionSimilarities, 'The Old Base Algorithm, just use SMILES code.')
    Fold_CNN_DDI = (412, Scenarios.FoldInteractionSimilarities, 'The Old Base Algorithm, use CNN.')
    Fold_KNN = (451, Scenarios.FoldInteractionSimilarities, 'This network learns by KNN.')
    Fold_SVM = (452, Scenarios.FoldInteractionSimilarities, 'This network learns by SVM.')
    Fold_LR = (453, Scenarios.FoldInteractionSimilarities, 'Logistic Regression.')
    Fold_RF = (454, Scenarios.FoldInteractionSimilarities, 'Random Forest.')

    Deep_DDI = (501, Scenarios.SplitInteractionSimilarities, 'The Old Base Algorithm, just use SMILES code.')
    DDIMDL = (502, Scenarios.SplitInteractionSimilarities, 'The Old Base Algorithm, just use SMILES code.')
    CNN_DDI = (503, Scenarios.SplitInteractionSimilarities, 'The Old Base Algorithm, use CNN.')

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
