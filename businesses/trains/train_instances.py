from businesses.trains.con_ae_dnn_train_service import ConAeDnnTrainService
from businesses.trains.deep_ddi_train_service import DeepDDITrainService
from businesses.trains.drug_ae_con_dnn_train_service import DrugAeConDnnTrainService
from businesses.trains.gat_ae_con_dnn_train_service import GatAeConDnnTrainService
from businesses.trains.lr_train_service import LrTrainService
from businesses.trains.mha_train_service import MhaTrainService
from businesses.trains.rf_train_service import RfTrainService
from businesses.trains.join_before_softmax_train_service import JoinBeforeSoftmaxTrainService
from businesses.trains.sum_softmax_outputs_train_service import SumSoftmaxOutputsTrainService
from businesses.trains.ae_con_dnn_train_service import AeConDnnTrainService
from businesses.trains.concat_dnn_train_service import ConcatDnnTrainService
from businesses.trains.knn_train_service import KnnTrainService
from businesses.trains.svm_train_service import SvmTrainService
from businesses.trains.train_base_service import TrainBaseService
from businesses.trains.train_plan_test import TrainPlanTest
from common.enums.train_models import TrainModel


def get_instance(category: TrainModel) -> TrainBaseService:
    if category == TrainModel.JoinBeforeSoftmax:
        return JoinBeforeSoftmaxTrainService(category)
    elif category == TrainModel.SumSoftmaxOutputs:
        return SumSoftmaxOutputsTrainService(category)
    elif category == TrainModel.AE_Con_DNN:
        return AeConDnnTrainService(category)
    elif category == TrainModel.Contact_DNN:
        return ConcatDnnTrainService(category)
    elif category == TrainModel.KNN:
        return KnnTrainService(category)
    elif category == TrainModel.SVM:
        return SvmTrainService(category)
    elif category == TrainModel.Con_AE_DNN:
        return ConAeDnnTrainService(category)
    elif category == TrainModel.LR:
        return LrTrainService(category)
    elif category == TrainModel.RF:
        return RfTrainService(category)
    elif category == TrainModel.GAT_AE_Con_DNN:
        return GatAeConDnnTrainService(category)
    elif category == TrainModel.MHA:
        return MhaTrainService(category)
    elif category == TrainModel.Drug_AE_Con_DNN:
        return DrugAeConDnnTrainService(category)
    elif category == TrainModel.Deep_DDI:
        return DeepDDITrainService(category)
    elif category == TrainModel.Test:
        return TrainPlanTest(category)
    else:
        raise ValueError("No suitable subclass found")
