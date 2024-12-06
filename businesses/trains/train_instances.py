from businesses.trains.split_drugs.drug_cnn_siam_train_service import DrugCNNSiamTrainService
from businesses.trains.split_drugs.drug_ddi_mdl_train_service import DrugDDIMDLTrainService
from businesses.trains.split_drugs.drug_deep_ddi_train_service import DrugDeepDDITrainService
from businesses.trains.split_drugs.drug_dm_ddi_train_service import DrugDMDDITrainService
from businesses.trains.split_drugs.drug_enc_con_dnn_train_service import DrugEncConDnnTrainService
from businesses.trains.split_drugs.drug_gat_ae_dnn_train_service import DrugGatAeDnnTrainService
from businesses.trains.split_drugs.drug_gat_enc_adv_dnn_train_service import DrugGatEncAdvDnnTrainService
from businesses.trains.split_drugs.drug_gat_enc_con_dnn_train_service import DrugGatEncConDnnTrainService
from businesses.trains.split_drugs.drug_gat_enc_mha_dnn_train_service import DrugGatEncMhaDnnTrainService
from businesses.trains.split_drugs.drug_gat_enc_sum_dnn_train_service import DrugGatEncSumDnnTrainService
from businesses.trains.split_drugs.drug_gat_enc_v2_dnn_train_service import DrugGatEncV2DnnTrainService
from businesses.trains.split_drugs.drug_gat_lstm_mha_dnn_train_service import DrugGatLstmMhaDnnTrainService
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
from businesses.trains.split_fold_interactions.fold_dm_ddi_train_service import FoldDMDDITrainService
from businesses.trains.split_fold_interactions.fold_gat_ae_dnn_train_service import FoldGatAeDnnTrainService
from businesses.trains.split_fold_interactions.fold_gat_enc_adv_dnn_train_service import FoldGatEncAdvDnnTrainService
from businesses.trains.split_fold_interactions.fold_gat_enc_con_dnn_train_service import FoldGatEncConDnnTrainService
from businesses.trains.split_fold_interactions.fold_gat_enc_mha_dnn_train_service import FoldGatEncMhaDnnTrainService
from businesses.trains.split_fold_interactions.fold_gat_enc_sum_dnn_train_service import FoldGatEncSumDnnTrainService
from businesses.trains.split_fold_interactions.fold_gat_enc_v2_dnn_train_service import FoldGatEncV2DnnTrainService
from businesses.trains.split_fold_interactions.fold_gat_lstm_mha_dnn_train_service import FoldGatLstmMhaDnnTrainService
from businesses.trains.split_fold_interactions.fold_gat_mha_dnn_train_service import FoldGatMhaDnnTrainService
from businesses.trains.split_fold_interactions.fold_gat_mha_rd_dnn_train_service import FoldGatMhaRDDnnTrainService
from businesses.trains.split_fold_interactions.fold_gat_mha_reverse_train_service import FoldGatMhaReverseTrainService
from businesses.trains.split_fold_interactions.fold_knn_train_service import FoldKnnTrainService
from businesses.trains.split_fold_interactions.fold_lr_train_service import FoldLrTrainService
from businesses.trains.split_fold_interactions.fold_rf_train_service import FoldRfTrainService
from businesses.trains.split_fold_interactions.fold_svm_train_service import FoldSvmTrainService
from businesses.trains.split_interactions.cnn_siam_train_service import CNNSiamTrainService
from businesses.trains.split_interactions.con_enc_dnn_train_service import ConEncDnnTrainService
from businesses.trains.split_interactions.concat_dnn_train_service import ConcatDnnTrainService
from businesses.trains.split_interactions.ddi_mdl_train_service import DDIMDLTrainService
from businesses.trains.split_interactions.deep_ddi_train_service import DeepDDITrainService
from businesses.trains.split_interactions.dm_ddi_train_service import DMDDITrainService
from businesses.trains.split_interactions.enc_con_dnn_train_service import EncConDnnTrainService
from businesses.trains.split_interactions.gat_ae_dnn_train_service import GatAeDnnTrainService
from businesses.trains.split_interactions.gat_enc_adv_dnn_train_service import GatEncAdvDnnTrainService
from businesses.trains.split_interactions.gat_enc_con_dnn_train_service import GatEncConDnnTrainService
from businesses.trains.split_interactions.gat_enc_mha_dnn_train_service import GatEncMhaDnnTrainService
from businesses.trains.split_interactions.gat_enc_sum_dnn_train_service import GatEncSumDnnTrainService
from businesses.trains.split_interactions.gat_enc_v2_dnn_train_service import GatEncV2DnnTrainService
from businesses.trains.split_interactions.gat_lstm_mha_dnn_train_service import GatLstmMhaDnnTrainService
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


drug_train_test_file_train_id = 0
drug_test_test_file_train_id = 0


def get_instance(category: TrainModel) -> TrainBaseService:
    if category == TrainModel.JoinBeforeSoftmax:
        return JoinBeforeSoftmaxTrainService(category)
    elif category == TrainModel.SumSoftmaxOutputs:
        return SumSoftmaxOutputsTrainService(category)
    elif category == TrainModel.Enc_Con_DNN:
        return EncConDnnTrainService(category)
    elif category == TrainModel.Contact_DNN:
        return ConcatDnnTrainService(category)
    elif category == TrainModel.DMDDI:
        return DMDDITrainService(category)
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
        return GatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3)
    elif category == TrainModel.GAT_Enc_Adv_DNN:
        return GatEncAdvDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[512, 256], droprate=0.3, pooling_mode='mean')
    elif category == TrainModel.GAT_Enc_Sum_DNN:
        return GatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3)
    elif category == TrainModel.GAT_Enc_V2:
        return GatEncV2DnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=8, dense_units=[512, 256], droprate=0.3, pooling_mode='mean')
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
    elif category == TrainModel.GAT_Lstm_MHA:
        return GatLstmMhaDnnTrainService(category)

    elif category == TrainModel.Drug_Enc_Con_DNN:
        return DrugEncConDnnTrainService(category, True, file_train_id=drug_train_test_file_train_id)
    elif category == TrainModel.Drug_Deep_DDI:
        return DrugDeepDDITrainService(category, True, file_train_id=drug_train_test_file_train_id)
    elif category == TrainModel.Drug_DDIMDL:
        return DrugDDIMDLTrainService(category, True, file_train_id=drug_train_test_file_train_id)
    elif category == TrainModel.Drug_CNN_Siam:
        return DrugCNNSiamTrainService(category, True, file_train_id=drug_train_test_file_train_id)
    elif category == TrainModel.Drug_GAT_Enc_Con_DNN:
        return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3,
                                            compare_train_test=True, file_train_id=drug_train_test_file_train_id)
    elif category == TrainModel.Drug_GAT_Enc_Adv_DNN:
        return DrugGatEncAdvDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[512, 256], droprate=0.3, pooling_mode='mean',
                                            compare_train_test=True, file_train_id=drug_train_test_file_train_id)
    elif category == TrainModel.Drug_GAT_Enc_Sum_DNN:
        return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3,
                                            compare_train_test=True, file_train_id=drug_train_test_file_train_id)
    elif category == TrainModel.Drug_GAT_Enc_V2:
        return DrugGatEncV2DnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=8, dense_units=[512, 256], droprate=0.3, pooling_mode='mean',
                                           compare_train_test=True, file_train_id=drug_train_test_file_train_id)
    elif category == TrainModel.Drug_GAT_MHA_DNN:
        return DrugGatMhaDnnTrainService(category, True, file_train_id=drug_train_test_file_train_id)
    elif category == TrainModel.Drug_GAT_AE_DNN:
        return DrugGatAeDnnTrainService(category, True, file_train_id=drug_train_test_file_train_id)
    elif category == TrainModel.Drug_GAT_MHA_Reverse:
        return DrugGatMhaReverseTrainService(category, True, file_train_id=drug_train_test_file_train_id)
    elif category == TrainModel.Drug_GAT_MHA_RD_DNN:
        return DrugGatMhaRDDnnTrainService(category, True, file_train_id=drug_train_test_file_train_id)
    elif category == TrainModel.Drug_GAT_Enc_MHA:
        return DrugGatEncMhaDnnTrainService(category, True, file_train_id=drug_train_test_file_train_id)
    elif category == TrainModel.Drug_GAT_Lstm_MHA:
        return DrugGatLstmMhaDnnTrainService(category, True, file_train_id=drug_train_test_file_train_id)
    elif category == TrainModel.Drug_DMDDI:
        return DrugDMDDITrainService(category, True, file_train_id=drug_train_test_file_train_id)
    elif category == TrainModel.Drug_KNN:
        return DrugKnnTrainService(category, True, file_train_id=drug_train_test_file_train_id)
    elif category == TrainModel.Drug_SVM:
        return DrugSvmTrainService(category, True, file_train_id=drug_train_test_file_train_id)
    elif category == TrainModel.Drug_LR:
        return DrugLrTrainService(category, True, file_train_id=drug_train_test_file_train_id)
    elif category == TrainModel.Drug_RF:
        return DrugRfTrainService(category, True, file_train_id=drug_train_test_file_train_id)

    elif category == TrainModel.Drug_Enc_Con_DNN_Test:
        return DrugEncConDnnTrainService(category, False, file_train_id=drug_test_test_file_train_id)
    elif category == TrainModel.Drug_Deep_DDI_Test:
        return DrugDeepDDITrainService(category, False, file_train_id=drug_test_test_file_train_id)
    elif category == TrainModel.Drug_DDIMDL_Test:
        return DrugDDIMDLTrainService(category, False, file_train_id=drug_test_test_file_train_id)
    elif category == TrainModel.Drug_CNN_Siam_Test:
        return DrugCNNSiamTrainService(category, False, file_train_id=drug_test_test_file_train_id)
    elif category == TrainModel.Drug_GAT_Enc_Con_DNN_Test:
        return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3,
                                            compare_train_test=False, file_train_id=drug_test_test_file_train_id)
    elif category == TrainModel.Drug_GAT_Enc_Adv_DNN_Test:
        return DrugGatEncAdvDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[512, 256], droprate=0.3, pooling_mode='mean',
                                            compare_train_test=False, file_train_id=drug_test_test_file_train_id)
    elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_Test:
        return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3,
                                            compare_train_test=False, file_train_id=drug_test_test_file_train_id)
    elif category == TrainModel.Drug_GAT_Enc_V2_Test:
        return DrugGatEncV2DnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=8, dense_units=[512, 256], droprate=0.3, pooling_mode='mean',
                                           compare_train_test=False, file_train_id=drug_test_test_file_train_id)
    elif category == TrainModel.Drug_GAT_MHA_DNN_Test:
        return DrugGatMhaDnnTrainService(category, False, file_train_id=drug_test_test_file_train_id)
    elif category == TrainModel.Drug_GAT_AE_DNN_Test:
        return DrugGatAeDnnTrainService(category, False, file_train_id=drug_test_test_file_train_id)
    elif category == TrainModel.Drug_GAT_MHA_Reverse_Test:
        return DrugGatMhaReverseTrainService(category, False, file_train_id=drug_test_test_file_train_id)
    elif category == TrainModel.Drug_GAT_MHA_RD_DNN_Test:
        return DrugGatMhaRDDnnTrainService(category, False, file_train_id=drug_test_test_file_train_id)
    elif category == TrainModel.Drug_GAT_Enc_MHA_Test:
        return DrugGatEncMhaDnnTrainService(category, False, file_train_id=drug_test_test_file_train_id)
    elif category == TrainModel.Drug_GAT_Lstm_MHA_Test:
        return DrugGatLstmMhaDnnTrainService(category, False, file_train_id=drug_test_test_file_train_id)
    elif category == TrainModel.Drug_DMDDI_Test:
        return DrugDMDDITrainService(category, False, file_train_id=drug_test_test_file_train_id)
    elif category == TrainModel.Drug_KNN_Test:
        return DrugKnnTrainService(category, False, file_train_id=drug_test_test_file_train_id)
    elif category == TrainModel.Drug_SVM_Test:
        return DrugSvmTrainService(category, False, file_train_id=drug_test_test_file_train_id)
    elif category == TrainModel.Drug_LR_Test:
        return DrugLrTrainService(category, False, file_train_id=drug_test_test_file_train_id)
    elif category == TrainModel.Drug_RF_Test:
        return DrugRfTrainService(category, False, file_train_id=drug_test_test_file_train_id)

    elif category == TrainModel.Fold_GAT_Enc_Con_DNN:
        return FoldGatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3)
    elif category == TrainModel.Fold_GAT_Enc_Adv_DNN:
        return FoldGatEncAdvDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[512, 256], droprate=0.3, pooling_mode='mean')
    elif category == TrainModel.Fold_GAT_Enc_Sum_DNN:
        return FoldGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3)
    elif category == TrainModel.Fold_GAT_Enc_V2:
        return FoldGatEncV2DnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=8, dense_units=[512, 256], droprate=0.3)
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
    elif category == TrainModel.Fold_GAT_Lstm_MHA:
        return FoldGatLstmMhaDnnTrainService(category)
    elif category == TrainModel.Fold_Deep_DDI:
        return FoldDeepDDITrainService(category)
    elif category == TrainModel.Fold_DDIMDL:
        return FoldDDIMDLTrainService(category)
    elif category == TrainModel.Fold_CNN_Siam:
        return FoldCNNSiamTrainService(category)
    elif category == TrainModel.Fold_DMDDI:
        return FoldDMDDITrainService(category)
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


    # elif category == TrainModel.GAT_Enc_Con_DNN_30:
    #     return GatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.GAT_Enc_Con_DNN_31:
    #     return GatEncConDnnTrainService(category, encoding_dim=64, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.GAT_Enc_Con_DNN_32:
    #     return GatEncConDnnTrainService(category, encoding_dim=256, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.GAT_Enc_Con_DNN_33:
    #     return GatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=4, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.GAT_Enc_Con_DNN_34:
    #     return GatEncConDnnTrainService(category, encoding_dim=128, gat_units=128, num_heads=4, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.GAT_Enc_Con_DNN_35:
    #     return GatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=8, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.GAT_Enc_Con_DNN_36:
    #     return GatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512, 256], droprate=0.3)
    # elif category == TrainModel.GAT_Enc_Con_DNN_37:
    #     return GatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256, 128], droprate=0.3)
    # elif category == TrainModel.GAT_Enc_Con_DNN_38:
    #     return GatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[256, 128], droprate=0.3)
    # elif category == TrainModel.GAT_Enc_Con_DNN_39:
    #     return GatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512], droprate=0.3)
    # elif category == TrainModel.GAT_Enc_Con_DNN_40:
    #     return GatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.2)
    # elif category == TrainModel.GAT_Enc_Con_DNN_41:
    #     return GatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.5)
    # elif category == TrainModel.GAT_Enc_Con_DNN_42:
    #     return GatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.GAT_Enc_Con_DNN_43:
    #     return GatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[1024, 512, 256], droprate=0.5)
    # elif category == TrainModel.GAT_Enc_Con_DNN_44:
    #     return GatEncConDnnTrainService(category, encoding_dim=128, gat_units=16, num_heads=4, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.GAT_Enc_Con_DNN_45:
    #     return GatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=16, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.GAT_Enc_Con_DNN_46:
    #     return GatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=16, dense_units=[1024, 512, 256], droprate=0.5)
    # elif category == TrainModel.GAT_Enc_Con_DNN_47:
    #     return GatEncConDnnTrainService(category, encoding_dim=256, gat_units=32, num_heads=16, dense_units=[1024, 512, 256], droprate=0.5)
    # elif category == TrainModel.GAT_Enc_Con_DNN_48:
    #     return GatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[512, 256], droprate=0.5, pooling_mode='mean')
    # elif category == TrainModel.GAT_Enc_Con_DNN_49:
    #     return GatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[512, 256], droprate=0.5, pooling_mode='max')
    # elif category == TrainModel.GAT_Enc_Con_DNN_50:
    #     return GatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[512, 256], droprate=0.5, pooling_mode='sum')
    # elif category == TrainModel.GAT_Enc_Con_DNN_51:
    #     return GatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[256, 128], droprate=0.5, pooling_mode='mean')
    # elif category == TrainModel.GAT_Enc_Con_DNN_52:
    #     return GatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[256, 128], droprate=0.5, pooling_mode='max')
    # elif category == TrainModel.GAT_Enc_Con_DNN_53:
    #     return GatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[256, 128], droprate=0.5, pooling_mode='sum')
    #
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_30:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_31:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=64, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3, compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_32:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=256, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_33:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=4, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_34:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=128, num_heads=4, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_35:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=8, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_36:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512, 256], droprate=0.3,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_37:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256, 128], droprate=0.3,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_38:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[256, 128], droprate=0.3,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_39:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512], droprate=0.3,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_40:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.2,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_41:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.5,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_42:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_43:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[1024, 512, 256], droprate=0.5,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_44:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=16, num_heads=4, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_45:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=16, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_46:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=16, dense_units=[1024, 512, 256], droprate=0.5,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_47:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=256, gat_units=32, num_heads=16, dense_units=[1024, 512, 256], droprate=0.5,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_48:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[512, 256], droprate=0.5, pooling_mode='mean',
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_49:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[512, 256], droprate=0.5, pooling_mode='max',
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_50:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[512, 256], droprate=0.5, pooling_mode='sum',
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_51:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[256, 128], droprate=0.5, pooling_mode='mean',
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_52:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[256, 128], droprate=0.5, pooling_mode='max',
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_53:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[256, 128], droprate=0.5, pooling_mode='sum',
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_Test_30:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_Test_31:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=64, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_Test_32:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=256, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_Test_33:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=4, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_Test_34:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=128, num_heads=4, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_Test_35:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=8, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_Test_36:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512, 256], droprate=0.3,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_Test_37:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256, 128], droprate=0.3,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_Test_38:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[256, 128], droprate=0.3,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_Test_39:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512], droprate=0.3,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_Test_40:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.2,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_Test_41:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.5,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_Test_42:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_Test_43:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[1024, 512, 256], droprate=0.5,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_Test_44:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=16, num_heads=4, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_Test_45:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=16, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_Test_46:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=16, dense_units=[1024, 512, 256], droprate=0.5,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_Test_47:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=256, gat_units=32, num_heads=16, dense_units=[1024, 512, 256], droprate=0.5,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_Test_48:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[512, 256], droprate=0.5, pooling_mode='mean',
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_Test_49:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[512, 256], droprate=0.5, pooling_mode='max',
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_Test_50:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[512, 256], droprate=0.5, pooling_mode='sum',
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_Test_51:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[256, 128], droprate=0.5, pooling_mode='mean',
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_Test_52:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[256, 128], droprate=0.5, pooling_mode='max',
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Con_DNN_Test_53:
    #     return DrugGatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[256, 128], droprate=0.5, pooling_mode='sum',
    #                                         compare_train_test=False)
    #
    # elif category == TrainModel.Fold_GAT_Enc_Con_DNN_30:
    #     return FoldGatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.Fold_GAT_Enc_Con_DNN_31:
    #     return FoldGatEncConDnnTrainService(category, encoding_dim=64, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.Fold_GAT_Enc_Con_DNN_32:
    #     return FoldGatEncConDnnTrainService(category, encoding_dim=256, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.Fold_GAT_Enc_Con_DNN_33:
    #     return FoldGatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=4, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.Fold_GAT_Enc_Con_DNN_34:
    #     return FoldGatEncConDnnTrainService(category, encoding_dim=128, gat_units=128, num_heads=4, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.Fold_GAT_Enc_Con_DNN_35:
    #     return FoldGatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=8, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.Fold_GAT_Enc_Con_DNN_36:
    #     return FoldGatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512, 256], droprate=0.3)
    # elif category == TrainModel.Fold_GAT_Enc_Con_DNN_37:
    #     return FoldGatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256, 128], droprate=0.3)
    # elif category == TrainModel.Fold_GAT_Enc_Con_DNN_38:
    #     return FoldGatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[256, 128], droprate=0.3)
    # elif category == TrainModel.Fold_GAT_Enc_Con_DNN_39:
    #     return FoldGatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512], droprate=0.3)
    # elif category == TrainModel.Fold_GAT_Enc_Con_DNN_40:
    #     return FoldGatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.2)
    # elif category == TrainModel.Fold_GAT_Enc_Con_DNN_41:
    #     return FoldGatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.5)
    # elif category == TrainModel.Fold_GAT_Enc_Con_DNN_42:
    #     return FoldGatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.Fold_GAT_Enc_Con_DNN_43:
    #     return FoldGatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[1024, 512, 256], droprate=0.5)
    # elif category == TrainModel.Fold_GAT_Enc_Con_DNN_44:
    #     return FoldGatEncConDnnTrainService(category, encoding_dim=128, gat_units=16, num_heads=4, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.Fold_GAT_Enc_Con_DNN_45:
    #     return FoldGatEncConDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=16, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.Fold_GAT_Enc_Con_DNN_46:
    #     return FoldGatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=16, dense_units=[1024, 512, 256], droprate=0.5)
    # elif category == TrainModel.Fold_GAT_Enc_Con_DNN_47:
    #     return FoldGatEncConDnnTrainService(category, encoding_dim=256, gat_units=32, num_heads=16, dense_units=[1024, 512, 256], droprate=0.5)
    # elif category == TrainModel.Fold_GAT_Enc_Con_DNN_48:
    #     return FoldGatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[512, 256], droprate=0.5, pooling_mode='mean')
    # elif category == TrainModel.Fold_GAT_Enc_Con_DNN_49:
    #     return FoldGatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[512, 256], droprate=0.5, pooling_mode='max')
    # elif category == TrainModel.Fold_GAT_Enc_Con_DNN_50:
    #     return FoldGatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[512, 256], droprate=0.5, pooling_mode='sum')
    # elif category == TrainModel.Fold_GAT_Enc_Con_DNN_51:
    #     return FoldGatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[256, 128], droprate=0.5, pooling_mode='mean')
    # elif category == TrainModel.Fold_GAT_Enc_Con_DNN_52:
    #     return FoldGatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[256, 128], droprate=0.5, pooling_mode='max')
    # elif category == TrainModel.Fold_GAT_Enc_Con_DNN_53:
    #     return FoldGatEncConDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[256, 128], droprate=0.5, pooling_mode='sum')
    #
    #
    # elif category == TrainModel.GAT_Enc_Sum_DNN_60:
    #     return GatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.GAT_Enc_Sum_DNN_61:
    #     return GatEncSumDnnTrainService(category, encoding_dim=64, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.GAT_Enc_Sum_DNN_62:
    #     return GatEncSumDnnTrainService(category, encoding_dim=256, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.GAT_Enc_Sum_DNN_63:
    #     return GatEncSumDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=4, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.GAT_Enc_Sum_DNN_64:
    #     return GatEncSumDnnTrainService(category, encoding_dim=128, gat_units=128, num_heads=4, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.GAT_Enc_Sum_DNN_65:
    #     return GatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=8, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.GAT_Enc_Sum_DNN_66:
    #     return GatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512, 256], droprate=0.3)
    # elif category == TrainModel.GAT_Enc_Sum_DNN_67:
    #     return GatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256, 128], droprate=0.3)
    # elif category == TrainModel.GAT_Enc_Sum_DNN_68:
    #     return GatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[256, 128], droprate=0.3)
    # elif category == TrainModel.GAT_Enc_Sum_DNN_69:
    #     return GatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512], droprate=0.3)
    # elif category == TrainModel.GAT_Enc_Sum_DNN_70:
    #     return GatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.2)
    # elif category == TrainModel.GAT_Enc_Sum_DNN_71:
    #     return GatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.5)
    # elif category == TrainModel.GAT_Enc_Sum_DNN_72:
    #     return GatEncSumDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.GAT_Enc_Sum_DNN_73:
    #     return GatEncSumDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[1024, 512, 256], droprate=0.5)
    # elif category == TrainModel.GAT_Enc_Sum_DNN_74:
    #     return GatEncSumDnnTrainService(category, encoding_dim=128, gat_units=16, num_heads=4, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.GAT_Enc_Sum_DNN_75:
    #     return GatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=16, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.GAT_Enc_Sum_DNN_76:
    #     return GatEncSumDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=16, dense_units=[1024, 512, 256], droprate=0.5)
    # elif category == TrainModel.GAT_Enc_Sum_DNN_77:
    #     return GatEncSumDnnTrainService(category, encoding_dim=256, gat_units=32, num_heads=16, dense_units=[1024, 512, 256], droprate=0.5)
    # elif category == TrainModel.GAT_Enc_Sum_DNN_78:
    #     return GatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.GAT_Enc_Sum_DNN_79:
    #     return GatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_60:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_61:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=64, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3, compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_62:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=256, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_63:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=4, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_64:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=128, num_heads=4, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_65:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=8, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_66:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512, 256], droprate=0.3,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_67:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256, 128], droprate=0.3,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_68:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[256, 128], droprate=0.3,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_69:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512], droprate=0.3,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_70:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.2,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_71:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.5,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_72:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_73:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[1024, 512, 256], droprate=0.5,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_74:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=16, num_heads=4, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_75:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=16, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_76:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=16, dense_units=[1024, 512, 256], droprate=0.5,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_77:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=256, gat_units=32, num_heads=16, dense_units=[1024, 512, 256], droprate=0.5,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_78:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_79:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_Test_60:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_Test_61:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=64, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_Test_62:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=256, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_Test_63:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=4, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_Test_64:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=128, num_heads=4, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_Test_65:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=8, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_Test_66:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512, 256], droprate=0.3,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_Test_67:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256, 128], droprate=0.3,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_Test_68:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[256, 128], droprate=0.3,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_Test_69:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512], droprate=0.3,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_Test_70:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.2,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_Test_71:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.5,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_Test_72:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_Test_73:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[1024, 512, 256], droprate=0.5,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_Test_74:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=16, num_heads=4, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_Test_75:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=16, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_Test_76:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=16, dense_units=[1024, 512, 256], droprate=0.5,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_Test_77:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=256, gat_units=32, num_heads=16, dense_units=[1024, 512, 256], droprate=0.5,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_Test_78:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Drug_GAT_Enc_Sum_DNN_Test_79:
    #     return DrugGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3,
    #                                         compare_train_test=False)
    # elif category == TrainModel.Fold_GAT_Enc_Sum_DNN_60:
    #     return FoldGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.Fold_GAT_Enc_Sum_DNN_61:
    #     return FoldGatEncSumDnnTrainService(category, encoding_dim=64, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.Fold_GAT_Enc_Sum_DNN_62:
    #     return FoldGatEncSumDnnTrainService(category, encoding_dim=256, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.Fold_GAT_Enc_Sum_DNN_63:
    #     return FoldGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=4, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.Fold_GAT_Enc_Sum_DNN_64:
    #     return FoldGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=128, num_heads=4, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.Fold_GAT_Enc_Sum_DNN_65:
    #     return FoldGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=8, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.Fold_GAT_Enc_Sum_DNN_66:
    #     return FoldGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512, 256], droprate=0.3)
    # elif category == TrainModel.Fold_GAT_Enc_Sum_DNN_67:
    #     return FoldGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256, 128], droprate=0.3)
    # elif category == TrainModel.Fold_GAT_Enc_Sum_DNN_68:
    #     return FoldGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[256, 128], droprate=0.3)
    # elif category == TrainModel.Fold_GAT_Enc_Sum_DNN_69:
    #     return FoldGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512], droprate=0.3)
    # elif category == TrainModel.Fold_GAT_Enc_Sum_DNN_70:
    #     return FoldGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.2)
    # elif category == TrainModel.Fold_GAT_Enc_Sum_DNN_71:
    #     return FoldGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.5)
    # elif category == TrainModel.Fold_GAT_Enc_Sum_DNN_72:
    #     return FoldGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.Fold_GAT_Enc_Sum_DNN_73:
    #     return FoldGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[1024, 512, 256], droprate=0.5)
    # elif category == TrainModel.Fold_GAT_Enc_Sum_DNN_74:
    #     return FoldGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=16, num_heads=4, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.Fold_GAT_Enc_Sum_DNN_75:
    #     return FoldGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=16, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.Fold_GAT_Enc_Sum_DNN_76:
    #     return FoldGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=16, dense_units=[1024, 512, 256], droprate=0.5)
    # elif category == TrainModel.Fold_GAT_Enc_Sum_DNN_77:
    #     return FoldGatEncSumDnnTrainService(category, encoding_dim=256, gat_units=32, num_heads=16, dense_units=[1024, 512, 256], droprate=0.5)
    # elif category == TrainModel.Fold_GAT_Enc_Sum_DNN_78:
    #     return FoldGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3)
    # elif category == TrainModel.Fold_GAT_Enc_Sum_DNN_79:
    #     return FoldGatEncSumDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[512, 256], droprate=0.3)


    elif category == TrainModel.Drug_GAT_Enc_Adv_DNN_30:
        return DrugGatEncAdvDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512, 256], droprate=0.3, pooling_mode='mean',
                                            batch_size=32, lr_rate=1e-4, adam_beta=[0.9, 0.999], alpha=0.0,
                                            compare_train_test=True)
    elif category == TrainModel.Drug_GAT_Enc_Adv_DNN_31:
        return DrugGatEncAdvDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512, 256], droprate=0.3, pooling_mode='mean',
                                            batch_size=64, lr_rate=1e-4, adam_beta=[0.9, 0.999], alpha=0.0,
                                            compare_train_test=True)
    elif category == TrainModel.Drug_GAT_Enc_Adv_DNN_32:
        return DrugGatEncAdvDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512, 256], droprate=0.3, pooling_mode='mean',
                                            batch_size=128, lr_rate=1e-4, adam_beta=[0.9, 0.999], alpha=0.0,
                                            compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Adv_DNN_33:
    #     return DrugGatEncAdvDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512, 256], droprate=0.3, pooling_mode='mean',
    #                                         batch_size=128, lr_rate=1e-6, adam_beta=[0.9, 0.999], alpha=0.0,
    #                                         compare_train_test=True)
    elif category == TrainModel.Drug_GAT_Enc_Adv_DNN_34:
        return DrugGatEncAdvDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512, 256], droprate=0.3, pooling_mode='mean',
                                            batch_size=128, lr_rate=1e-4, adam_beta=[0.85, 0.995], alpha=0.0,
                                            compare_train_test=True)
    elif category == TrainModel.Drug_GAT_Enc_Adv_DNN_35:
        return DrugGatEncAdvDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512, 256], droprate=0.3, pooling_mode='mean',
                                            batch_size=128, lr_rate=1e-4, adam_beta=[0.95, 0.9995], alpha=0.0,
                                            compare_train_test=True)
    elif category == TrainModel.Drug_GAT_Enc_Adv_DNN_36:
        return DrugGatEncAdvDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512, 256], droprate=0.3, pooling_mode='mean',
                                            batch_size=128, lr_rate=1e-4, adam_beta=[0.9, 0.999], alpha=0.0,
                                            compare_train_test=True)
    elif category == TrainModel.Drug_GAT_Enc_Adv_DNN_37:
        return DrugGatEncAdvDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512, 256], droprate=0.3, pooling_mode='mean',
                                            batch_size=128, lr_rate=1e-4, adam_beta=[0.9, 0.999], alpha=0.1,
                                            compare_train_test=True)
    elif category == TrainModel.Drug_GAT_Enc_Adv_DNN_38:
        return DrugGatEncAdvDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512, 256], droprate=0.3, pooling_mode='mean',
                                            batch_size=128, lr_rate=1e-4, adam_beta=[0.9, 0.999], alpha=0.0, schedule_number=2,
                                            compare_train_test=True)
    elif category == TrainModel.Drug_GAT_Enc_Adv_DNN_39:
        return DrugGatEncAdvDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512, 256], droprate=0.3, pooling_mode='mean',
                                            batch_size=128, lr_rate=1e-3, adam_beta=[0.9, 0.999], alpha=0.0,
                                            compare_train_test=True)
    elif category == TrainModel.Drug_GAT_Enc_Adv_DNN_40:
        return DrugGatEncAdvDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512, 256], droprate=0.3, pooling_mode='mean',
                                            batch_size=256, lr_rate=1e-4, adam_beta=[0.9, 0.999], alpha=0.0,
                                            compare_train_test=True)
    elif category == TrainModel.Drug_GAT_Enc_Adv_DNN_41:
        return DrugGatEncAdvDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512, 256], droprate=0.3, pooling_mode='mean',
                                            batch_size=256, lr_rate=1e-4, adam_beta=[0.9, 0.999], alpha=0.1,
                                            compare_train_test=True)
    elif category == TrainModel.Drug_GAT_Enc_Adv_DNN_42:
        return DrugGatEncAdvDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512, 256], droprate=0.3, pooling_mode='mean',
                                            batch_size=256, lr_rate=1e-4, adam_beta=[0.9, 0.999], alpha=0.1, schedule_number=2,
                                            compare_train_test=True)
    elif category == TrainModel.Drug_GAT_Enc_Adv_DNN_43:
        return DrugGatEncAdvDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512, 256], droprate=0.3, pooling_mode='mean',
                                            batch_size=64, lr_rate=1e-4, adam_beta=[0.9, 0.999], alpha=0.1,
                                            compare_train_test=True)
    elif category == TrainModel.Drug_GAT_Enc_Adv_DNN_44:
        return DrugGatEncAdvDnnTrainService(category, encoding_dim=128, gat_units=64, num_heads=4, dense_units=[1024, 512, 256], droprate=0.3, pooling_mode='mean',
                                            batch_size=64, lr_rate=1e-4, adam_beta=[0.9, 0.999], alpha=0.1, schedule_number=2,
                                            compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Adv_DNN_45:
    #     return DrugGatEncAdvDnnTrainService(category, ,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Adv_DNN_46:
    #     return DrugGatEncAdvDnnTrainService(category, ,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Adv_DNN_47:
    #     return DrugGatEncAdvDnnTrainService(category, encoding_dim=256, gat_units=32, num_heads=16, dense_units=[1024, 512, 256], droprate=0.5, pooling_mode='mean',
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Adv_DNN_48:
    #     return DrugGatEncAdvDnnTrainService(category, ,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Adv_DNN_49:
    #     return DrugGatEncAdvDnnTrainService(category, ,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Adv_DNN_50:
    #     return DrugGatEncAdvDnnTrainService(category, ,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Adv_DNN_51:
    #     return DrugGatEncAdvDnnTrainService(category, encoding_dim=128, gat_units=32, num_heads=8, dense_units=[2048, 1024, 512, 256], droprate=0.5, pooling_mode='mean',
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Adv_DNN_52:
    #     return DrugGatEncAdvDnnTrainService(category, ,
    #                                         compare_train_test=True)
    # elif category == TrainModel.Drug_GAT_Enc_Adv_DNN_53:
    #     return DrugGatEncAdvDnnTrainService(category, ,
    #                                         compare_train_test=True)


    else:
        raise ValueError("No suitable subclass found")

    # DOI 10.3389/fphar.2024.1354540
