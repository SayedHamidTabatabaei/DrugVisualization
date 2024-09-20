from enum import Enum


class ComparePlotType(Enum):
    Accuracy = (1, 'accuracy', 'bar')
    Loss = (2, 'loss', 'bar')
    F1_Score = (3, 'f1_score', 'bar')
    AUC = (4, 'auc', 'bar')
    AUPR = (5, 'aupr', 'bar')
    Recall = (6, 'recall', 'bar')
    Precision = (7, 'precision', 'bar')
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
