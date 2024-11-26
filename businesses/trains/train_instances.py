from businesses.trains.split_drugs.drug_cnn_siam_train_service import DrugCNNSiamTrainService
from businesses.trains.split_drugs.drug_gat_ae_dnn_train_service import DrugGatAeDnnTrainService
from businesses.trains.split_drugs.drug_gat_enc_mha_dnn_train_service import DrugGatEncMhaDnnTrainService
from businesses.trains.split_drugs.drug_gat_mha_dnn_train_service import DrugGatMhaDnnTrainService
from businesses.trains.split_drugs.drug_gat_mha_rd_dnn_train_service import DrugGatMhaRDDnnTrainService
from businesses.trains.split_drugs.drug_gat_mha_reverse_train_service import DrugGatMhaReverseTrainService
from businesses.trains.split_drugs.drug_knn_train_service import DrugKnnTrainService
from businesses.trains.split_drugs.drug_lr_train_service import DrugLrTrainService
from businesses.trains.split_drugs.drug_rf_train_service import DrugRfTrainService
from businesses.trains.split_drugs.drug_svm_train_service import DrugSvmTrainService
from businesses.trains.split_fold_interactions.fold_cnn_siam_train_service import FoldCNNSiamTrainService
from businesses.trains.split_fold_interactions.fold_ddi_mdl_train_service import FoldDDIMDLTrainService
from businesses.trains.split_fold_interactions.fold_deep_ddi_train_service import FoldDeepDDITrainService
from businesses.trains.split_fold_interactions.fold_gat_ae_dnn_train_service import FoldGatAeDnnTrainService
from businesses.trains.split_fold_interactions.fold_gat_enc_con_dnn_train_service import FoldGatEncConDnnTrainService
from businesses.trains.split_fold_interactions.fold_gat_enc_mha_dnn_train_service import FoldGatEncMhaDnnTrainService
from businesses.trains.split_fold_interactions.fold_gat_mha_dnn_train_service import FoldGatMhaDnnTrainService
from businesses.trains.split_fold_interactions.fold_gat_mha_rd_dnn_train_service import FoldGatMhaRDDnnTrainService
from businesses.trains.split_fold_interactions.fold_gat_mha_reverse_train_service import FoldGatMhaReverseTrainService
from businesses.trains.split_fold_interactions.fold_knn_train_service import FoldKnnTrainService
from businesses.trains.split_fold_interactions.fold_lr_train_service import FoldLrTrainService
from businesses.trains.split_fold_interactions.fold_rf_train_service import FoldRfTrainService
from businesses.trains.split_fold_interactions.fold_svm_train_service import FoldSvmTrainService
from businesses.trains.split_interactions.enc_con_dnn_train_service import EncConDnnTrainService
from businesses.trains.split_interactions.cnn_siam_train_service import CNNSiamTrainService
from businesses.trains.split_interactions.con_enc_dnn_train_service import ConEncDnnTrainService
from businesses.trains.split_interactions.concat_dnn_train_service import ConcatDnnTrainService
from businesses.trains.split_interactions.ddi_mdl_train_service import DDIMDLTrainService
from businesses.trains.split_interactions.deep_ddi_train_service import DeepDDITrainService
from businesses.trains.split_drugs.drug_enc_con_dnn_train_service import DrugEncConDnnTrainService
from businesses.trains.split_drugs.drug_ddi_mdl_train_service import DrugDDIMDLTrainService
from businesses.trains.split_drugs.drug_deep_ddi_train_service import DrugDeepDDITrainService
from businesses.trains.split_drugs.drug_gat_enc_con_dnn_train_service import DrugGatEncConDnnTrainService
from businesses.trains.split_interactions.gat_ae_dnn_train_service import GatAeDnnTrainService
from businesses.trains.split_interactions.gat_enc_con_dnn_train_service import GatEncConDnnTrainService
from businesses.trains.split_interactions.gat_enc_mha_dnn_train_service import GatEncMhaDnnTrainService
from businesses.trains.split_interactions.gat_mha_dnn_train_service import GatMhaDnnTrainService
from businesses.trains.split_interactions.gat_mha_rd_dnn_train_service import GatMhaRDDnnTrainService
from businesses.trains.split_interactions.gat_mha_reverse_train_service import GatMhaReverseTrainService
from businesses.trains.split_interactions.join_before_softmax_train_service import JoinBeforeSoftmaxTrainService
from businesses.trains.split_interactions.knn_train_service import KnnTrainService
from businesses.trains.split_interactions.lr_train_service import LrTrainService
from businesses.trains.split_interactions.rf_train_service import RfTrainService
from businesses.trains.split_interactions.sum_softmax_outputs_train_service import SumSoftmaxOutputsTrainService
from businesses.trains.split_interactions.svm_train_service import SvmTrainService
from businesses.trains.train_base_service import TrainBaseService
from common.enums.train_models import TrainModel


def get_instance(category: TrainModel) -> TrainBaseService:
    if category == TrainModel.JoinBeforeSoftmax:
        return JoinBeforeSoftmaxTrainService(category)
    elif category == TrainModel.SumSoftmaxOutputs:
        return SumSoftmaxOutputsTrainService(category)
    elif category == TrainModel.Enc_Con_DNN:
        return EncConDnnTrainService(category)
    elif category == TrainModel.Contact_DNN:
        return ConcatDnnTrainService(category)
    elif category == TrainModel.KNN:
        return KnnTrainService(category)
    elif category == TrainModel.SVM:
        return SvmTrainService(category)
    elif category == TrainModel.Con_AE_DNN:
        return ConEncDnnTrainService(category)
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
    elif category == TrainModel.GAT_MHA_Reverse:
        return GatMhaReverseTrainService(category)
    elif category == TrainModel.GAT_MHA_RD_DNN:
        return GatMhaRDDnnTrainService(category)
    elif category == TrainModel.GAT_Enc_MHA:
        return GatEncMhaDnnTrainService(category)

    elif category == TrainModel.Drug_Enc_Con_DNN:
        return DrugEncConDnnTrainService(category, True)
    elif category == TrainModel.Drug_Deep_DDI:
        return DrugDeepDDITrainService(category, True)
    elif category == TrainModel.Drug_DDIMDL:
        return DrugDDIMDLTrainService(category, True)
    elif category == TrainModel.Drug_CNN_Siam:
        return DrugCNNSiamTrainService(category, True)
    elif category == TrainModel.Drug_GAT_Enc_Con_DNN:
        return DrugGatEncConDnnTrainService(category, True)
    elif category == TrainModel.Drug_GAT_MHA_DNN:
        return DrugGatMhaDnnTrainService(category, True)
    elif category == TrainModel.Drug_GAT_AE_DNN:
        return DrugGatAeDnnTrainService(category, True)
    elif category == TrainModel.Drug_GAT_MHA_Reverse:
        return DrugGatMhaReverseTrainService(category, True)
    elif category == TrainModel.Drug_GAT_MHA_RD_DNN:
        return DrugGatMhaRDDnnTrainService(category, True)
    elif category == TrainModel.Drug_GAT_Enc_MHA:
        return DrugGatEncMhaDnnTrainService(category, True)
    elif category == TrainModel.Drug_KNN:
        return DrugKnnTrainService(category, True)
    elif category == TrainModel.Drug_SVM:
        return DrugSvmTrainService(category, True)
    elif category == TrainModel.Drug_LR:
        return DrugLrTrainService(category, True)
    elif category == TrainModel.Drug_RF:
        return DrugRfTrainService(category, True)

    elif category == TrainModel.Drug_Enc_Con_DNN_Test:
        return DrugEncConDnnTrainService(category, False)
    elif category == TrainModel.Drug_Deep_DDI_Test:
        return DrugDeepDDITrainService(category, False)
    elif category == TrainModel.Drug_DDIMDL_Test:
        return DrugDDIMDLTrainService(category, False)
    elif category == TrainModel.Drug_CNN_Siam_Test:
        return DrugCNNSiamTrainService(category, False)
    elif category == TrainModel.Drug_GAT_Enc_Con_DNN_Test:
        return DrugGatEncConDnnTrainService(category, False)
    elif category == TrainModel.Drug_GAT_MHA_DNN_Test:
        return DrugGatMhaDnnTrainService(category, False)
    elif category == TrainModel.Drug_GAT_AE_DNN_Test:
        return DrugGatAeDnnTrainService(category, False)
    elif category == TrainModel.Drug_GAT_MHA_Reverse_Test:
        return DrugGatMhaReverseTrainService(category, False)
    elif category == TrainModel.Drug_GAT_MHA_RD_DNN_Test:
        return DrugGatMhaRDDnnTrainService(category, False)
    elif category == TrainModel.Drug_GAT_Enc_MHA_Test:
        return DrugGatEncMhaDnnTrainService(category, False)
    elif category == TrainModel.Drug_KNN_Test:
        return DrugKnnTrainService(category, False)
    elif category == TrainModel.Drug_SVM_Test:
        return DrugSvmTrainService(category, False)
    elif category == TrainModel.Drug_LR_Test:
        return DrugLrTrainService(category, False)
    elif category == TrainModel.Drug_RF_Test:
        return DrugRfTrainService(category, False)

    elif category == TrainModel.Fold_GAT_Enc_Con_DNN:
        return FoldGatEncConDnnTrainService(category)
    elif category == TrainModel.Fold_GAT_MHA_DNN:
        return FoldGatMhaDnnTrainService(category)
    elif category == TrainModel.Fold_GAT_AE_DNN:
        return FoldGatAeDnnTrainService(category)
    elif category == TrainModel.Fold_GAT_MHA_Reverse:
        return FoldGatMhaReverseTrainService(category)
    elif category == TrainModel.Fold_GAT_MHA_RD_DNN:
        return FoldGatMhaRDDnnTrainService(category)
    elif category == TrainModel.Fold_GAT_Enc_MHA:
        return FoldGatEncMhaDnnTrainService(category)
    elif category == TrainModel.Fold_Deep_DDI:
        return FoldDeepDDITrainService(category)
    elif category == TrainModel.Fold_DDIMDL:
        return FoldDDIMDLTrainService(category)
    elif category == TrainModel.Fold_CNN_Siam:
        return FoldCNNSiamTrainService(category)
    elif category == TrainModel.Fold_KNN:
        return FoldKnnTrainService(category)
    elif category == TrainModel.Fold_SVM:
        return FoldSvmTrainService(category)
    elif category == TrainModel.Fold_LR:
        return FoldLrTrainService(category)
    elif category == TrainModel.Fold_RF:
        return FoldRfTrainService(category)

    elif category == TrainModel.Deep_DDI:
        return DeepDDITrainService(category)
    elif category == TrainModel.DDIMDL:
        return DDIMDLTrainService(category)
    elif category == TrainModel.CNN_Siam:
        return CNNSiamTrainService(category)
    else:
        raise ValueError("No suitable subclass found")
