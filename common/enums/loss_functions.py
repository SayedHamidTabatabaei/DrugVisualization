from enum import Enum

from common.enums.train_models import TrainModel


class LossFunctions(Enum):
    categorical_crossentropy = (1, "categorical_crossentropy", "categorical cross entropy=",
                                [TrainModel.Enc_Con_DNN, TrainModel.Drug_Enc_Con_DNN, TrainModel.Drug_Enc_Con_DNN_Test,
                                 TrainModel.Con_AE_DNN,
                                 TrainModel.Contact_DNN,
                                 TrainModel.JoinBeforeSoftmax,
                                 TrainModel.SumSoftmaxOutputs,
                                 TrainModel.GAT_Enc_Con_DNN, TrainModel.Drug_GAT_Enc_Con_DNN, TrainModel.Drug_GAT_Enc_Con_DNN_Test, TrainModel.Fold_GAT_Enc_Con_DNN,
                                 TrainModel.GAT_MHA_DNN, TrainModel.Drug_GAT_MHA_DNN, TrainModel.Drug_GAT_MHA_DNN_Test, TrainModel.Fold_GAT_MHA_DNN,
                                 TrainModel.GAT_AE_DNN, TrainModel.Drug_GAT_AE_DNN, TrainModel.Drug_GAT_AE_DNN_Test, TrainModel.Fold_GAT_AE_DNN,
                                 TrainModel.GAT_MHA_RD_DNN, TrainModel.Drug_GAT_MHA_RD_DNN, TrainModel.Drug_GAT_MHA_RD_DNN_Test, TrainModel.Fold_GAT_MHA_RD_DNN,
                                 TrainModel.GAT_Enc_MHA, TrainModel.Drug_GAT_Enc_MHA, TrainModel.Drug_GAT_Enc_MHA_Test, TrainModel.Fold_GAT_Enc_MHA,
                                 TrainModel.GAT_Enc_Sum_DNN, TrainModel.Drug_GAT_Enc_Sum_DNN, TrainModel.Drug_GAT_Enc_Sum_DNN_Test, TrainModel.Fold_GAT_Enc_Sum_DNN,
                                 TrainModel.GAT_Enc_Con_DNN_30, TrainModel.Drug_GAT_Enc_Con_DNN_30, TrainModel.Drug_GAT_Enc_Con_DNN_Test_30, TrainModel.Fold_GAT_Enc_Con_DNN_30,
                                 TrainModel.GAT_Enc_Con_DNN_31, TrainModel.Drug_GAT_Enc_Con_DNN_31, TrainModel.Drug_GAT_Enc_Con_DNN_Test_31, TrainModel.Fold_GAT_Enc_Con_DNN_31,
                                 TrainModel.GAT_Enc_Con_DNN_32, TrainModel.Drug_GAT_Enc_Con_DNN_32, TrainModel.Drug_GAT_Enc_Con_DNN_Test_32, TrainModel.Fold_GAT_Enc_Con_DNN_32,
                                 TrainModel.GAT_Enc_Con_DNN_33, TrainModel.Drug_GAT_Enc_Con_DNN_33, TrainModel.Drug_GAT_Enc_Con_DNN_Test_33, TrainModel.Fold_GAT_Enc_Con_DNN_33,
                                 TrainModel.GAT_Enc_Con_DNN_34, TrainModel.Drug_GAT_Enc_Con_DNN_34, TrainModel.Drug_GAT_Enc_Con_DNN_Test_34, TrainModel.Fold_GAT_Enc_Con_DNN_34,
                                 TrainModel.GAT_Enc_Con_DNN_35, TrainModel.Drug_GAT_Enc_Con_DNN_35, TrainModel.Drug_GAT_Enc_Con_DNN_Test_35, TrainModel.Fold_GAT_Enc_Con_DNN_35,
                                 TrainModel.GAT_Enc_Con_DNN_36, TrainModel.Drug_GAT_Enc_Con_DNN_36, TrainModel.Drug_GAT_Enc_Con_DNN_Test_36, TrainModel.Fold_GAT_Enc_Con_DNN_36,
                                 TrainModel.GAT_Enc_Con_DNN_37, TrainModel.Drug_GAT_Enc_Con_DNN_37, TrainModel.Drug_GAT_Enc_Con_DNN_Test_37, TrainModel.Fold_GAT_Enc_Con_DNN_37,
                                 TrainModel.GAT_Enc_Con_DNN_38, TrainModel.Drug_GAT_Enc_Con_DNN_38, TrainModel.Drug_GAT_Enc_Con_DNN_Test_38, TrainModel.Fold_GAT_Enc_Con_DNN_38,
                                 TrainModel.GAT_Enc_Con_DNN_39, TrainModel.Drug_GAT_Enc_Con_DNN_39, TrainModel.Drug_GAT_Enc_Con_DNN_Test_39, TrainModel.Fold_GAT_Enc_Con_DNN_39,
                                 TrainModel.GAT_Enc_Con_DNN_40, TrainModel.Drug_GAT_Enc_Con_DNN_40, TrainModel.Drug_GAT_Enc_Con_DNN_Test_40, TrainModel.Fold_GAT_Enc_Con_DNN_40,
                                 TrainModel.GAT_Enc_Con_DNN_41, TrainModel.Drug_GAT_Enc_Con_DNN_41, TrainModel.Drug_GAT_Enc_Con_DNN_Test_41, TrainModel.Fold_GAT_Enc_Con_DNN_41,
                                 TrainModel.GAT_Enc_Con_DNN_42, TrainModel.Drug_GAT_Enc_Con_DNN_42, TrainModel.Drug_GAT_Enc_Con_DNN_Test_42, TrainModel.Fold_GAT_Enc_Con_DNN_42,
                                 TrainModel.GAT_Enc_Con_DNN_43, TrainModel.Drug_GAT_Enc_Con_DNN_43, TrainModel.Drug_GAT_Enc_Con_DNN_Test_43, TrainModel.Fold_GAT_Enc_Con_DNN_43,
                                 TrainModel.GAT_Enc_Con_DNN_44, TrainModel.Drug_GAT_Enc_Con_DNN_44, TrainModel.Drug_GAT_Enc_Con_DNN_Test_44, TrainModel.Fold_GAT_Enc_Con_DNN_44,
                                 TrainModel.GAT_Enc_Con_DNN_45, TrainModel.Drug_GAT_Enc_Con_DNN_45, TrainModel.Drug_GAT_Enc_Con_DNN_Test_45, TrainModel.Fold_GAT_Enc_Con_DNN_45,
                                 TrainModel.GAT_Enc_Con_DNN_46, TrainModel.Drug_GAT_Enc_Con_DNN_46, TrainModel.Drug_GAT_Enc_Con_DNN_Test_46, TrainModel.Fold_GAT_Enc_Con_DNN_46,
                                 TrainModel.GAT_Enc_Con_DNN_47, TrainModel.Drug_GAT_Enc_Con_DNN_47, TrainModel.Drug_GAT_Enc_Con_DNN_Test_47, TrainModel.Fold_GAT_Enc_Con_DNN_47,
                                 TrainModel.GAT_Enc_Con_DNN_48, TrainModel.Drug_GAT_Enc_Con_DNN_48, TrainModel.Drug_GAT_Enc_Con_DNN_Test_48, TrainModel.Fold_GAT_Enc_Con_DNN_48,
                                 TrainModel.GAT_Enc_Con_DNN_49, TrainModel.Drug_GAT_Enc_Con_DNN_49, TrainModel.Drug_GAT_Enc_Con_DNN_Test_49, TrainModel.Fold_GAT_Enc_Con_DNN_49,
                                 TrainModel.GAT_Enc_Sum_DNN_60, TrainModel.Drug_GAT_Enc_Sum_DNN_60, TrainModel.Drug_GAT_Enc_Sum_DNN_Test_60, TrainModel.Fold_GAT_Enc_Sum_DNN_60,
                                 TrainModel.GAT_Enc_Sum_DNN_61, TrainModel.Drug_GAT_Enc_Sum_DNN_61, TrainModel.Drug_GAT_Enc_Sum_DNN_Test_61, TrainModel.Fold_GAT_Enc_Sum_DNN_61,
                                 TrainModel.GAT_Enc_Sum_DNN_62, TrainModel.Drug_GAT_Enc_Sum_DNN_62, TrainModel.Drug_GAT_Enc_Sum_DNN_Test_62, TrainModel.Fold_GAT_Enc_Sum_DNN_62,
                                 TrainModel.GAT_Enc_Sum_DNN_63, TrainModel.Drug_GAT_Enc_Sum_DNN_63, TrainModel.Drug_GAT_Enc_Sum_DNN_Test_63, TrainModel.Fold_GAT_Enc_Sum_DNN_63,
                                 TrainModel.GAT_Enc_Sum_DNN_64, TrainModel.Drug_GAT_Enc_Sum_DNN_64, TrainModel.Drug_GAT_Enc_Sum_DNN_Test_64, TrainModel.Fold_GAT_Enc_Sum_DNN_64,
                                 TrainModel.GAT_Enc_Sum_DNN_65, TrainModel.Drug_GAT_Enc_Sum_DNN_65, TrainModel.Drug_GAT_Enc_Sum_DNN_Test_65, TrainModel.Fold_GAT_Enc_Sum_DNN_65,
                                 TrainModel.GAT_Enc_Sum_DNN_66, TrainModel.Drug_GAT_Enc_Sum_DNN_66, TrainModel.Drug_GAT_Enc_Sum_DNN_Test_66, TrainModel.Fold_GAT_Enc_Sum_DNN_66,
                                 TrainModel.GAT_Enc_Sum_DNN_67, TrainModel.Drug_GAT_Enc_Sum_DNN_67, TrainModel.Drug_GAT_Enc_Sum_DNN_Test_67, TrainModel.Fold_GAT_Enc_Sum_DNN_67,
                                 TrainModel.GAT_Enc_Sum_DNN_68, TrainModel.Drug_GAT_Enc_Sum_DNN_68, TrainModel.Drug_GAT_Enc_Sum_DNN_Test_68, TrainModel.Fold_GAT_Enc_Sum_DNN_68,
                                 TrainModel.GAT_Enc_Sum_DNN_69, TrainModel.Drug_GAT_Enc_Sum_DNN_69, TrainModel.Drug_GAT_Enc_Sum_DNN_Test_69, TrainModel.Fold_GAT_Enc_Sum_DNN_69,
                                 TrainModel.GAT_Enc_Sum_DNN_70, TrainModel.Drug_GAT_Enc_Sum_DNN_70, TrainModel.Drug_GAT_Enc_Sum_DNN_Test_70, TrainModel.Fold_GAT_Enc_Sum_DNN_70,
                                 TrainModel.GAT_Enc_Sum_DNN_71, TrainModel.Drug_GAT_Enc_Sum_DNN_71, TrainModel.Drug_GAT_Enc_Sum_DNN_Test_71, TrainModel.Fold_GAT_Enc_Sum_DNN_71,
                                 TrainModel.GAT_Enc_Sum_DNN_72, TrainModel.Drug_GAT_Enc_Sum_DNN_72, TrainModel.Drug_GAT_Enc_Sum_DNN_Test_72, TrainModel.Fold_GAT_Enc_Sum_DNN_72,
                                 TrainModel.GAT_Enc_Sum_DNN_73, TrainModel.Drug_GAT_Enc_Sum_DNN_73, TrainModel.Drug_GAT_Enc_Sum_DNN_Test_73, TrainModel.Fold_GAT_Enc_Sum_DNN_73,
                                 TrainModel.GAT_Enc_Sum_DNN_74, TrainModel.Drug_GAT_Enc_Sum_DNN_74, TrainModel.Drug_GAT_Enc_Sum_DNN_Test_74, TrainModel.Fold_GAT_Enc_Sum_DNN_74,
                                 TrainModel.GAT_Enc_Sum_DNN_75, TrainModel.Drug_GAT_Enc_Sum_DNN_75, TrainModel.Drug_GAT_Enc_Sum_DNN_Test_75, TrainModel.Fold_GAT_Enc_Sum_DNN_75,
                                 TrainModel.GAT_Enc_Sum_DNN_76, TrainModel.Drug_GAT_Enc_Sum_DNN_76, TrainModel.Drug_GAT_Enc_Sum_DNN_Test_76, TrainModel.Fold_GAT_Enc_Sum_DNN_76,
                                 TrainModel.GAT_Enc_Sum_DNN_77, TrainModel.Drug_GAT_Enc_Sum_DNN_77, TrainModel.Drug_GAT_Enc_Sum_DNN_Test_77, TrainModel.Fold_GAT_Enc_Sum_DNN_77,
                                 TrainModel.GAT_Enc_Sum_DNN_78, TrainModel.Drug_GAT_Enc_Sum_DNN_78, TrainModel.Drug_GAT_Enc_Sum_DNN_Test_78, TrainModel.Fold_GAT_Enc_Sum_DNN_78,
                                 TrainModel.GAT_Enc_Sum_DNN_79, TrainModel.Drug_GAT_Enc_Sum_DNN_79, TrainModel.Drug_GAT_Enc_Sum_DNN_Test_79, TrainModel.Fold_GAT_Enc_Sum_DNN_79,
                                 ])
    sparse_categorical_crossentropy = (1, "sparse_categorical_crossentropy", "sparse categorical cross entropy=",
                                       [TrainModel.Enc_Con_DNN, TrainModel.Drug_Enc_Con_DNN, TrainModel.Drug_Enc_Con_DNN_Test,
                                        TrainModel.Con_AE_DNN,
                                        TrainModel.Contact_DNN,
                                        TrainModel.JoinBeforeSoftmax,
                                        TrainModel.SumSoftmaxOutputs,
                                        TrainModel.GAT_Enc_Con_DNN, TrainModel.Drug_GAT_Enc_Con_DNN, TrainModel.Drug_GAT_Enc_Con_DNN_Test, TrainModel.Fold_GAT_Enc_Con_DNN,
                                        TrainModel.GAT_MHA_DNN, TrainModel.Drug_GAT_MHA_DNN, TrainModel.Drug_GAT_MHA_DNN_Test, TrainModel.Fold_GAT_MHA_DNN,
                                        TrainModel.GAT_AE_DNN, TrainModel.Drug_GAT_AE_DNN, TrainModel.Drug_GAT_AE_DNN_Test, TrainModel.Fold_GAT_AE_DNN,
                                        TrainModel.GAT_MHA_RD_DNN, TrainModel.Drug_GAT_MHA_RD_DNN, TrainModel.Drug_GAT_MHA_RD_DNN_Test, TrainModel.Fold_GAT_MHA_RD_DNN,
                                        TrainModel.GAT_Enc_MHA, TrainModel.Drug_GAT_Enc_MHA, TrainModel.Drug_GAT_Enc_MHA_Test, TrainModel.Fold_GAT_Enc_MHA
                                        ])
    focal = (3, "Focal Loss", "Focal Loss=",
             [TrainModel.Enc_Con_DNN, TrainModel.Drug_Enc_Con_DNN, TrainModel.Drug_Enc_Con_DNN_Test,
              TrainModel.Con_AE_DNN,
              TrainModel.Contact_DNN,
              TrainModel.JoinBeforeSoftmax,
              TrainModel.SumSoftmaxOutputs,
              TrainModel.GAT_Enc_Con_DNN, TrainModel.Drug_GAT_Enc_Con_DNN, TrainModel.Drug_GAT_Enc_Con_DNN_Test, TrainModel.Fold_GAT_Enc_Con_DNN,
              TrainModel.GAT_MHA_DNN, TrainModel.Drug_GAT_MHA_DNN, TrainModel.Drug_GAT_MHA_DNN_Test, TrainModel.Fold_GAT_MHA_DNN,
              TrainModel.GAT_AE_DNN, TrainModel.Drug_GAT_AE_DNN, TrainModel.Drug_GAT_AE_DNN_Test, TrainModel.Fold_GAT_AE_DNN,
              TrainModel.GAT_MHA_RD_DNN, TrainModel.Drug_GAT_MHA_RD_DNN, TrainModel.Drug_GAT_MHA_RD_DNN_Test, TrainModel.Fold_GAT_MHA_RD_DNN,
              TrainModel.GAT_Enc_MHA, TrainModel.Drug_GAT_Enc_MHA, TrainModel.Drug_GAT_Enc_MHA_Test, TrainModel.Fold_GAT_Enc_MHA])
    focal_tversky = (4, "Focal Tversky Loss", "Focal Tversky Loss=",
                     [TrainModel.Enc_Con_DNN, TrainModel.Drug_Enc_Con_DNN, TrainModel.Drug_Enc_Con_DNN_Test,
                      TrainModel.Con_AE_DNN,
                      TrainModel.Contact_DNN,
                      TrainModel.JoinBeforeSoftmax,
                      TrainModel.SumSoftmaxOutputs,
                      TrainModel.GAT_Enc_Con_DNN, TrainModel.Drug_GAT_Enc_Con_DNN, TrainModel.Drug_GAT_Enc_Con_DNN_Test, TrainModel.Fold_GAT_Enc_Con_DNN,
                      TrainModel.GAT_MHA_DNN, TrainModel.Drug_GAT_MHA_DNN, TrainModel.Drug_GAT_MHA_DNN_Test, TrainModel.Fold_GAT_MHA_DNN,
                      TrainModel.GAT_AE_DNN, TrainModel.Drug_GAT_AE_DNN, TrainModel.Drug_GAT_AE_DNN_Test, TrainModel.Fold_GAT_AE_DNN,
                      TrainModel.GAT_MHA_RD_DNN, TrainModel.Drug_GAT_MHA_RD_DNN, TrainModel.Drug_GAT_MHA_RD_DNN_Test, TrainModel.Fold_GAT_MHA_RD_DNN,
                      TrainModel.GAT_Enc_MHA, TrainModel.Drug_GAT_Enc_MHA, TrainModel.Drug_GAT_Enc_MHA_Test, TrainModel.Fold_GAT_Enc_MHA])
    squared_hinge = (5, "squared_hinge", "squared hinge=",
                     [TrainModel.Enc_Con_DNN, TrainModel.Drug_Enc_Con_DNN, TrainModel.Drug_Enc_Con_DNN_Test,
                      TrainModel.Con_AE_DNN,
                      TrainModel.Contact_DNN,
                      TrainModel.JoinBeforeSoftmax,
                      TrainModel.SumSoftmaxOutputs,
                      TrainModel.GAT_Enc_Con_DNN, TrainModel.Drug_GAT_Enc_Con_DNN, TrainModel.Drug_GAT_Enc_Con_DNN_Test, TrainModel.Fold_GAT_Enc_Con_DNN,
                      TrainModel.GAT_MHA_DNN, TrainModel.Drug_GAT_MHA_DNN, TrainModel.Drug_GAT_MHA_DNN_Test, TrainModel.Fold_GAT_MHA_DNN,
                      TrainModel.GAT_AE_DNN, TrainModel.Drug_GAT_AE_DNN, TrainModel.Drug_GAT_AE_DNN_Test, TrainModel.Fold_GAT_AE_DNN,
                      TrainModel.GAT_MHA_RD_DNN, TrainModel.Drug_GAT_MHA_RD_DNN, TrainModel.Drug_GAT_MHA_RD_DNN_Test, TrainModel.Fold_GAT_MHA_RD_DNN,
                      TrainModel.GAT_Enc_MHA, TrainModel.Drug_GAT_Enc_MHA, TrainModel.Drug_GAT_Enc_MHA_Test, TrainModel.Fold_GAT_Enc_MHA])
    dice_loss = (6, "dice", "dice=",
                 [TrainModel.Enc_Con_DNN, TrainModel.Drug_Enc_Con_DNN, TrainModel.Drug_Enc_Con_DNN_Test,
                  TrainModel.Con_AE_DNN,
                  TrainModel.Contact_DNN,
                  TrainModel.JoinBeforeSoftmax,
                  TrainModel.SumSoftmaxOutputs,
                  TrainModel.GAT_Enc_Con_DNN, TrainModel.Drug_GAT_Enc_Con_DNN, TrainModel.Drug_GAT_Enc_Con_DNN_Test, TrainModel.Fold_GAT_Enc_Con_DNN,
                  TrainModel.GAT_MHA_DNN, TrainModel.Drug_GAT_MHA_DNN, TrainModel.Drug_GAT_MHA_DNN_Test, TrainModel.Fold_GAT_MHA_DNN,
                  TrainModel.GAT_AE_DNN, TrainModel.Drug_GAT_AE_DNN, TrainModel.Drug_GAT_AE_DNN_Test, TrainModel.Fold_GAT_AE_DNN,
                  TrainModel.GAT_MHA_RD_DNN, TrainModel.Drug_GAT_MHA_RD_DNN, TrainModel.Drug_GAT_MHA_RD_DNN_Test, TrainModel.Fold_GAT_MHA_RD_DNN,
                  TrainModel.GAT_Enc_MHA, TrainModel.Drug_GAT_Enc_MHA, TrainModel.Drug_GAT_Enc_MHA_Test, TrainModel.Fold_GAT_Enc_MHA
                  ])

    def __init__(self, value, display_name, formula, valid_train_models):
        self._value_ = value
        self.display_name = display_name
        self.formula = formula
        self.valid_train_models = valid_train_models

    @classmethod
    def valid(cls, train_model: TrainModel):
        return list({loss_function for loss_function in cls if train_model in loss_function.valid_train_models})

    @classmethod
    def from_value(cls, value):
        return next((lf for lf in cls if lf.value == value), None)
