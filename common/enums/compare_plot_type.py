from enum import Enum


class ComparePlotType(Enum):
    Accuracy = (1, 'accuracy', 'bar')
    Loss = (2, 'loss', 'bar')
    F1_Score_Weighted = (3, 'f1_score_weighted', 'bar')
    F1_Score_Micro = (4, 'f1_score_micro', 'bar')
    F1_Score_Macro = (5, 'f1_score_macro', 'bar')
    AUC_Weighted = (4, 'auc_weighted', 'bar')
    AUC_Micro = (4, 'auc_micro', 'bar')
    AUC_Macro = (4, 'auc_macro', 'bar')
    AUPR_Weighted = (5, 'aupr_weighted', 'bar')
    AUPR_Micro = (5, 'aupr_micro', 'bar')
    AUPR_Macro = (5, 'aupr_macro', 'bar')
    Recall_Weighted = (6, 'recall_weighted', 'bar')
    Recall_Micro = (6, 'recall_micro', 'bar')
    Recall_Macro = (6, 'recall_macro', 'bar')
    Precision_Weighted = (7, 'precision_weighted', 'bar')
    Precision_Micro = (7, 'precision_micro', 'bar')
    Precision_Macro = (7, 'precision_macro', 'bar')
    Details_Accuracy = (8, 'details_accuracy', 'radial')
    Details_F1_Score = (9, 'details_f1_score', 'radial')
    Details_AUC = (10, 'details_auc', 'radial')
    Details_AUPR = (11, 'details_aupr', 'radial')
    Details_Recall = (10, 'details_recall', 'radial')
    Details_Precision = (11, 'details_precision', 'radial')

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
