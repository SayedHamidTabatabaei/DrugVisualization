from enum import Enum


class TrainingResultType(Enum):
    accuracy = (1, 'Accuracy')
    loss = (2, 'Loss')
    f1_score_weighted = (3, 'F1 Score - Weighted')
    f1_score_micro = (4, 'F1 Score - Micro')
    f1_score_macro = (5, 'F1 Score - Macro')
    auc_weighted = (6, 'AUC - Weighted')
    auc_micro = (7, 'AUC - Micro')
    auc_macro = (8, 'AUC - Macro')
    aupr_weighted = (9, 'AUPR - Weighted')
    aupr_micro = (10, 'AUPR - Micro')
    aupr_macro = (11, 'AUPR - Macro')
    recall_weighted = (12, 'Recall - Weighted')
    recall_micro = (13, 'Recall - Micro')
    recall_macro = (14, 'Recall - Macro')
    precision_weighted = (15, 'Precision - Weighted')
    precision_micro = (16, 'Precision - Micro')
    precision_macro = (17, 'Precision - Macro')

    def __init__(self, value, display_name):
        self._value_ = value
        self.display_name = display_name
