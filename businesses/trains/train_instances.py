from businesses.trains.split_interactions.ae_con_dnn_train_service import AeConDnnTrainService
from businesses.trains.split_interactions.con_ae_dnn_train_service import ConAeDnnTrainService
from businesses.trains.split_interactions.concat_dnn_train_service import ConcatDnnTrainService
from businesses.trains.split_interactions.ddi_mdl_train_service import DDIMDLTrainService
from businesses.trains.split_interactions.deep_ddi_train_service import DeepDDITrainService
from businesses.trains.split_drugs.drug_ae_con_dnn_train_service import DrugAeConDnnTrainService
from businesses.trains.split_drugs.drug_ddi_mdl_train_service import DrugDDIMDLTrainService
from businesses.trains.split_drugs.drug_deep_ddi_train_service import DrugDeepDDITrainService
from businesses.trains.split_drugs.drug_gat_enc_con_dnn_train_service import DrugGatEncConDnnTrainService
from businesses.trains.split_interactions.gat_ae_dnn_train_service import GatAeDnnTrainService
from businesses.trains.split_interactions.gat_enc_con_dnn_train_service import GatEncConDnnTrainService
from businesses.trains.split_interactions.gat_mha_dnn_train_service import GatMhaDnnTrainService
from businesses.trains.split_interactions.join_before_softmax_train_service import JoinBeforeSoftmaxTrainService
from businesses.trains.split_interactions.knn_train_service import KnnTrainService
from businesses.trains.split_interactions.lr_train_service import LrTrainService
from businesses.trains.split_interactions.rf_train_service import RfTrainService
from businesses.trains.split_interactions.sum_softmax_outputs_train_service import SumSoftmaxOutputsTrainService
from businesses.trains.split_interactions.svm_train_service import SvmTrainService
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
    elif category == TrainModel.GAT_Enc_Con_DNN:
        return GatEncConDnnTrainService(category)
    elif category == TrainModel.GAT_MHA_DNN:
        return GatMhaDnnTrainService(category)
    elif category == TrainModel.GAT_AE_DNN:
        return GatAeDnnTrainService(category)
    elif category == TrainModel.Drug_AE_Con_DNN:
        return DrugAeConDnnTrainService(category, True)
    elif category == TrainModel.Drug_DDIMDL:
        return DrugDDIMDLTrainService(category, True)
    elif category == TrainModel.Drug_Deep_DDI:
        return DrugDeepDDITrainService(category, True)
    elif category == TrainModel.Drug_GAT_Enc_Con_DNN:
        return DrugGatEncConDnnTrainService(category, True)
    elif category == TrainModel.Drug_AE_Con_DNN_Test:
        return DrugAeConDnnTrainService(category, False)
    elif category == TrainModel.Drug_DDIMDL_Test:
        return DrugDDIMDLTrainService(category, False)
    elif category == TrainModel.Drug_Deep_DDI_Test:
        return DrugDeepDDITrainService(category, False)
    elif category == TrainModel.Drug_GAT_Enc_Con_DNN_Test:
        return DrugGatEncConDnnTrainService(category, False)
    elif category == TrainModel.Deep_DDI:
        return DeepDDITrainService(category)
    elif category == TrainModel.DDIMDL:
        return DDIMDLTrainService(category)
    elif category == TrainModel.Test:
        return TrainPlanTest(category)
    else:
        raise ValueError("No suitable subclass found")
