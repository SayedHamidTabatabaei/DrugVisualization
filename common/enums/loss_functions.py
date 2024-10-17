from enum import Enum

from common.enums.train_models import TrainModel


class LossFunctions(Enum):
    categorical_crossentropy = (1, "categorical_crossentropy", "categorical cross entropy=",
                                [TrainModel.AE_Con_DNN, TrainModel.Con_AE_DNN, TrainModel.Contact_DNN, TrainModel.JoinBeforeSoftmax,
                                 TrainModel.SumSoftmaxOutputs, TrainModel.MHA, TrainModel.GAT_AE_Con_DNN, TrainModel.GAT_Con_AE_DNN])
    sparse_categorical_crossentropy = (1, "sparse_categorical_crossentropy", "sparse categorical cross entropy=",
                                       [TrainModel.AE_Con_DNN, TrainModel.Con_AE_DNN, TrainModel.Contact_DNN, TrainModel.JoinBeforeSoftmax,
                                        TrainModel.SumSoftmaxOutputs, TrainModel.MHA, TrainModel.GAT_AE_Con_DNN, TrainModel.GAT_Con_AE_DNN])
    focal = (3, "Focal Loss", "Focal Loss=",
             [TrainModel.AE_Con_DNN, TrainModel.Con_AE_DNN, TrainModel.Contact_DNN, TrainModel.JoinBeforeSoftmax,
              TrainModel.SumSoftmaxOutputs, TrainModel.MHA, TrainModel.GAT_AE_Con_DNN, TrainModel.GAT_Con_AE_DNN])
    focal_tversky = (4, "Focal Tversky Loss", "Focal Tversky Loss=",
                     [TrainModel.AE_Con_DNN, TrainModel.Con_AE_DNN, TrainModel.Contact_DNN, TrainModel.JoinBeforeSoftmax,
                      TrainModel.SumSoftmaxOutputs, TrainModel.MHA, TrainModel.GAT_AE_Con_DNN, TrainModel.GAT_Con_AE_DNN])
    squared_hinge = (5, "squared_hinge", "squared hinge=",
                     [TrainModel.AE_Con_DNN, TrainModel.Con_AE_DNN, TrainModel.Contact_DNN, TrainModel.JoinBeforeSoftmax,
                      TrainModel.SumSoftmaxOutputs, TrainModel.MHA, TrainModel.GAT_AE_Con_DNN, TrainModel.GAT_Con_AE_DNN])
    dice_loss = (6, "dice", "dice=",
                 [TrainModel.AE_Con_DNN, TrainModel.Con_AE_DNN, TrainModel.Contact_DNN, TrainModel.JoinBeforeSoftmax,
                  TrainModel.SumSoftmaxOutputs, TrainModel.MHA, TrainModel.GAT_AE_Con_DNN, TrainModel.GAT_Con_AE_DNN])

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
