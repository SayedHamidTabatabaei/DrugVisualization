from enum import Enum


class ComparePlotType(Enum):
    Accuracy = (1, 'accuracy', 'bar', 'Accuracy', '')
    Loss = (2, 'loss', 'bar', 'Loss', '')
    F1_Score_Weighted = (3, 'f1_score_weighted', 'bar', 'F1-Score', 'Weighted')
    F1_Score_Micro = (4, 'f1_score_micro', 'bar', 'F1-Score', 'Micro')
    F1_Score_Macro = (5, 'f1_score_macro', 'bar', 'F1-Score', 'Macro')
    AUC_Weighted = (6, 'auc_weighted', 'bar', 'AUC', 'Weighted')
    AUC_Micro = (7, 'auc_micro', 'bar', 'AUC', 'Micro')
    AUC_Macro = (8, 'auc_macro', 'bar', 'AUC', 'Macro')
    AUPR_Weighted = (9, 'aupr_weighted', 'bar', 'AUPR', 'Weighted')
    AUPR_Micro = (10, 'aupr_micro', 'bar', 'AUPR', 'Micro')
    AUPR_Macro = (11, 'aupr_macro', 'bar', 'AUPR', 'Macro')
    Recall_Weighted = (12, 'recall_weighted', 'bar', 'Recall', 'Weighted')
    Recall_Micro = (13, 'recall_micro', 'bar', 'Recall', 'Micro')
    Recall_Macro = (14, 'recall_macro', 'bar', 'Recall', 'Macro')
    Precision_Weighted = (15, 'precision_weighted', 'bar', 'Precision', 'Weighted')
    Precision_Micro = (16, 'precision_micro', 'bar', 'Precision', 'Micro')
    Precision_Macro = (17, 'precision_macro', 'bar', 'Precision', 'Macro')
    Details_Accuracy = (18, 'details_accuracy', 'radial', 'Accuracy', '')
    Details_F1_Score = (19, 'details_f1_score', 'radial', 'F1-Score', '')
    Details_AUC = (20, 'details_auc', 'radial', 'AUC', '')
    Details_AUPR = (21, 'details_aupr', 'radial', 'AUPR', '')
    Details_Recall = (22, 'details_recall', 'radial', 'Recall', '')
    Details_Precision = (23, 'details_precision', 'radial', 'Precision', '')

    def __init__(self, value, display_name, plot_name, title, second_title):
        self._value_ = value
        self.display_name = display_name
        self.plot_name = plot_name
        self.title = title
        self.second_title = second_title

    @staticmethod
    def get_enum_from_string(value):
        for member in ComparePlotType:
            if member.name.lower() == value.lower():
                return member
        raise ValueError(f"No matching enum found for display_name: {value}")
