from enum import Enum


class ComparePlotType(Enum):
    Accuracy = (1, 'accuracy')
    Loss = (2, 'loss')
    AUC = (3, 'auc')
    AUPR = (4, 'aupr')
    F1_Score = (5, 'f1_score')
    Details_Accuracy = (6, 'details_accuracy')
    Details_AUC = (7, 'details_auc')
    Details_AUPR = (8, 'details_aupr')
    Details_F1_Score = (9, 'details_f1_score')

    def __init__(self, value, display_name):
        self._value_ = value
        self.display_name = display_name

    @staticmethod
    def get_enum_from_string(value):
        for member in ComparePlotType:
            if member.name.lower() == value.lower():
                return member
        raise ValueError(f"No matching enum found for display_name: {value}")
