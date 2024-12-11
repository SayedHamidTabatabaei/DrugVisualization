from enum import Enum


class ComparePlotType(Enum):
    Accuracy = (1, 'accuracy', 'bar')
    Loss = (2, 'loss', 'bar')
    F1_Score_Weighted = (3, 'f1_score_weighted', 'bar')
    F1_Score_Micro = (4, 'f1_score_micro', 'bar')
    F1_Score_Macro = (5, 'f1_score_macro', 'bar')
    AUC_Weighted = (6, 'auc_weighted', 'bar')
    AUC_Micro = (7, 'auc_micro', 'bar')
    AUC_Macro = (8, 'auc_macro', 'bar')
    AUPR_Weighted = (9, 'aupr_weighted', 'bar')
    AUPR_Micro = (10, 'aupr_micro', 'bar')
    AUPR_Macro = (11, 'aupr_macro', 'bar')
    Recall_Weighted = (12, 'recall_weighted', 'bar')
    Recall_Micro = (13, 'recall_micro', 'bar')
    Recall_Macro = (14, 'recall_macro', 'bar')
    Precision_Weighted = (15, 'precision_weighted', 'bar')
    Precision_Micro = (16, 'precision_micro', 'bar')
    Precision_Macro = (17, 'precision_macro', 'bar')
    Details_Accuracy = (18, 'details_accuracy', 'radial')
    Details_F1_Score = (19, 'details_f1_score', 'radial')
    Details_AUC = (20, 'details_auc', 'radial')
    Details_AUPR = (21, 'details_aupr', 'radial')
    Details_Recall = (22, 'details_recall', 'radial')
    Details_Precision = (23, 'details_precision', 'radial')

    def __init__(self, value, display_name, plot_name):
        self._value_ = value
        self.display_name = display_name
        self.plot_name = plot_name

    @staticmethod
    def get_enum_from_string(value):
        for member in ComparePlotType:
            if member.name.lower() == value.lower():
                return member
        raise ValueError(f"No matching enum found for display_name: {value}")
