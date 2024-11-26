from enum import Enum

from common.enums.scenarios import Scenarios


class TrainModel(Enum):
    JoinBeforeSoftmax = (2, Scenarios.SplitInteractionSimilarities, 'NotFound', 'Having n separate network, but before softmax layer, join the last layer.')
    SumSoftmaxOutputs = (3, Scenarios.SplitInteractionSimilarities, 'NotFound', 'Have n separate network, and finally sum them.')
    Enc_Con_DNN = (4, Scenarios.SplitInteractionSimilarities, 'NotFound', 'Reduce data dimension, join them and send to DNN')
    Contact_DNN = (5, Scenarios.SplitInteractionSimilarities, 'NotFound', 'Just join all data and send to DNN.')
    KNN = (6, Scenarios.SplitInteractionSimilarities, 'NotFound', 'This network learns by KNN.')
    KNNWithAutoEncoder = (7, Scenarios.SplitInteractionSimilarities, 'NotFound', 'This network learns by KNN and reduce dimension by AutoEncoder.')
    SVM = (8, Scenarios.SplitInteractionSimilarities, 'NotFound', 'This network learns by SVM.')
    Con_AE_DNN = (9, Scenarios.SplitInteractionSimilarities, 'NotFound', 'This network learns by Concat input data AutoEncoder and then DNN.')
    LR = (10, Scenarios.SplitInteractionSimilarities, 'NotFound', 'Logistic Regression.')
    RF = (11, Scenarios.SplitInteractionSimilarities, 'NotFound', 'Random Forest.')
    GAT_Enc_Con_DNN = (12, Scenarios.SplitInteractionSimilarities, 'NotFound', 'GAT On SMILES, Encoder, Concat AutoEncoders and then DNN')
    GAT_MHA_DNN = (13, Scenarios.SplitInteractionSimilarities, 'NotFound', 'GAT On SMILES, Multi-Head-Attention between text fields and other fields (Pathway, '
                                                                           'Target, Enzyme) and then DNN.')
    GAT_AE_DNN = (14, Scenarios.SplitInteractionSimilarities, 'NotFound', 'GAT On SMILES, AutoEncoder, Concat AutoEncoders and then DNN')
    GAT_MHA_Reverse = (15, Scenarios.SplitInteractionSimilarities, 'NotFound', 'GAT On SMILES, Multi-Head-Attention between text fields and other fields ('
                                                                               'Pathway, Target, Enzyme) and then DNN.')
    GAT_MHA_RD_DNN = (16, Scenarios.SplitInteractionSimilarities, 'NotFound', 'GAT On SMILES, Multi-Head-Attention between text fields and other fields ('
                                                                              'Pathway, Target, Enzyme) and then DNN.')
    GAT_Enc_MHA = (20, Scenarios.SplitInteractionSimilarities, 'NotFound', 'GAT On SMILES, Multi-Head-Attention between text fields and other fields ('
                                                                           'Pathway, Target, Enzyme) and then DNN.')

    Drug_Enc_Con_DNN = (204, Scenarios.SplitDrugsTestWithTrain, 'NotFound', 'Reduce data dimension, join them and send to DNN')
    Drug_GAT_Enc_Con_DNN = (205, Scenarios.SplitDrugsTestWithTrain, 'NotFound', 'GAT On SMILES, AutoEncoder, Concat AutoEncoders and then DNN')
    Drug_GAT_MHA_DNN = (206, Scenarios.SplitDrugsTestWithTrain, 'NotFound', 'GAT On SMILES, Multi-Head-Attention between text fields and other fields ('
                                                                            'Pathway, Target, Enzyme) and then DNN.')
    Drug_GAT_AE_DNN = (207, Scenarios.SplitDrugsTestWithTrain, 'NotFound', 'GAT On SMILES, AutoEncoder, Concat AutoEncoders and then DNN')
    Drug_GAT_MHA_Reverse = (208, Scenarios.SplitDrugsTestWithTrain, 'NotFound', 'GAT On SMILES, Multi-Head-Attention between text fields and other fields ('
                                                                                'Pathway, Target, Enzyme) and then DNN.')
    Drug_GAT_MHA_RD_DNN = (209, Scenarios.SplitDrugsTestWithTrain, 'NotFound', 'GAT On SMILES, Multi-Head-Attention between text fields and other fields ('
                                                                               'Pathway, Target, Enzyme) and then DNN.')

    Drug_Deep_DDI = (210, Scenarios.SplitDrugsTestWithTrain, 'Deep_DDI', 'The Old Base Algorithm, just use SMILES code.')
    Drug_DDIMDL = (211, Scenarios.SplitDrugsTestWithTrain, 'DDIMDL', 'The Old Base Algorithm, just use SMILES code.')
    Drug_CNN_Siam = (212, Scenarios.SplitDrugsTestWithTrain, 'CNN_Siam', 'The Old Base Algorithm, use CNN.')
    Drug_GAT_Enc_MHA = (220, Scenarios.SplitDrugsTestWithTrain, 'NotFound', 'GAT On SMILES, Multi-Head-Attention between text fields and other fields ('
                                                                            'Pathway, Target, Enzyme) and then DNN.')

    Drug_KNN = (251, Scenarios.SplitDrugsTestWithTrain, 'NotFound', 'This network learns by KNN.')
    Drug_SVM = (252, Scenarios.SplitDrugsTestWithTrain, 'NotFound', 'This network learns by SVM.')
    Drug_LR = (253, Scenarios.SplitDrugsTestWithTrain, 'NotFound', 'Logistic Regression.')
    Drug_RF = (254, Scenarios.SplitDrugsTestWithTrain, 'NotFound', 'Random Forest.')

    Drug_Enc_Con_DNN_Test = (304, Scenarios.SplitDrugsTestWithTest, 'NotFound', 'Reduce data dimension, join them and send to DNN')
    Drug_GAT_Enc_Con_DNN_Test = (305, Scenarios.SplitDrugsTestWithTest, 'NotFound', 'GAT On SMILES, AutoEncoder, Concat AutoEncoders and then DNN')
    Drug_GAT_MHA_DNN_Test = (306, Scenarios.SplitDrugsTestWithTest, 'NotFound', 'GAT On SMILES, Multi-Head-Attention between text fields and other fields ('
                                                                                'Pathway, Target, Enzyme) and then DNN.')
    Drug_GAT_AE_DNN_Test = (307, Scenarios.SplitDrugsTestWithTest, 'NotFound', 'GAT On SMILES, AutoEncoder, Concat AutoEncoders and then DNN')
    Drug_GAT_MHA_Reverse_Test = (308, Scenarios.SplitDrugsTestWithTest, 'NotFound', 'GAT On SMILES, Multi-Head-Attention between text fields and other fields ('
                                                                                    'Pathway, Target, Enzyme) and then DNN.')
    Drug_GAT_MHA_RD_DNN_Test = (309, Scenarios.SplitDrugsTestWithTest, 'NotFound', 'GAT On SMILES, Multi-Head-Attention between text fields and other fields ('
                                                                                   'Pathway, Target, Enzyme) and then DNN.')
    Drug_Deep_DDI_Test = (310, Scenarios.SplitDrugsTestWithTest, 'Deep_DDI', 'The Old Base Algorithm, just use SMILES code.')
    Drug_DDIMDL_Test = (311, Scenarios.SplitDrugsTestWithTest, 'DDIMDL', 'The Old Base Algorithm, just use SMILES code.')
    Drug_CNN_Siam_Test = (312, Scenarios.SplitDrugsTestWithTest, 'CNN_Siam', 'The Old Base Algorithm, use CNN.')
    Drug_GAT_Enc_MHA_Test = (320, Scenarios.SplitDrugsTestWithTest, 'NotFound', 'GAT On SMILES, Multi-Head-Attention between text fields and other fields ('
                                                                                'Pathway, Target, Enzyme) and then DNN.')

    Drug_KNN_Test = (351, Scenarios.SplitDrugsTestWithTest, 'NotFound', 'This network learns by KNN.')
    Drug_SVM_Test = (352, Scenarios.SplitDrugsTestWithTest, 'NotFound', 'This network learns by SVM.')
    Drug_LR_Test = (353, Scenarios.SplitDrugsTestWithTest, 'NotFound', 'Logistic Regression.')
    Drug_RF_Test = (354, Scenarios.SplitDrugsTestWithTest, 'NotFound', 'Random Forest.')

    Fold_GAT_Enc_Con_DNN = (405, Scenarios.FoldInteractionSimilarities, 'NotFound', 'GAT On SMILES, AutoEncoder, Concat AutoEncoders and then DNN')
    Fold_GAT_MHA_DNN = (406, Scenarios.FoldInteractionSimilarities, 'NotFound', 'GAT On SMILES, Multi-Head-Attention between text fields and other fields ('
                                                                                'Pathway, Target, Enzyme) and then DNN.')
    Fold_GAT_AE_DNN = (407, Scenarios.FoldInteractionSimilarities, 'NotFound', 'GAT On SMILES, AutoEncoder, Concat AutoEncoders and then DNN')
    Fold_GAT_MHA_Reverse = (408, Scenarios.FoldInteractionSimilarities, 'NotFound', 'GAT On SMILES, Multi-Head-Attention between text fields and other fields ('
                                                                                    'Pathway, Target, Enzyme) and then DNN.')
    Fold_GAT_MHA_RD_DNN = (409, Scenarios.FoldInteractionSimilarities, 'NotFound', 'GAT On SMILES, Multi-Head-Attention between text fields and other fields ('
                                                                                   'Pathway, Target, Enzyme) and then DNN.')
    Fold_Deep_DDI = (410, Scenarios.FoldInteractionSimilarities, 'Deep_DDI', 'The Old Base Algorithm, just use SMILES code.')
    Fold_DDIMDL = (411, Scenarios.FoldInteractionSimilarities, 'DDIMDL', 'The Old Base Algorithm, just use SMILES code.')
    Fold_CNN_Siam = (412, Scenarios.FoldInteractionSimilarities, 'CNN_Siam', 'The Old Base Algorithm, use CNN.')
    Fold_GAT_Enc_MHA = (420, Scenarios.FoldInteractionSimilarities, 'NotFound', 'GAT On SMILES, Multi-Head-Attention between text fields and other fields ('
                                                                                'Pathway, Target, Enzyme) and then DNN.')

    Fold_KNN = (451, Scenarios.FoldInteractionSimilarities, 'NotFound', 'This network learns by KNN.')
    Fold_SVM = (452, Scenarios.FoldInteractionSimilarities, 'NotFound', 'This network learns by SVM.')
    Fold_LR = (453, Scenarios.FoldInteractionSimilarities, 'NotFound', 'Logistic Regression.')
    Fold_RF = (454, Scenarios.FoldInteractionSimilarities, 'NotFound', 'Random Forest.')

    Deep_DDI = (501, Scenarios.SplitInteractionSimilarities, 'Deep_DDI', 'The Old Base Algorithm, just use SMILES code.')
    DDIMDL = (502, Scenarios.SplitInteractionSimilarities, 'DDIMDL', 'The Old Base Algorithm, just use SMILES code.')
    CNN_Siam = (503, Scenarios.SplitInteractionSimilarities, 'CNN_Siam', 'The Old Base Algorithm, use CNN.')

    def __init__(self, value, scenario: Scenarios, image_name, description):
        self._value_ = value
        self.scenario = scenario
        self.image_name = image_name
        self.description = description

    @classmethod
    def from_value(cls, value):
        for item in cls:
            if item.value == value:
                return item
        raise ValueError(f"{value} is not a valid {cls.__name__}")
