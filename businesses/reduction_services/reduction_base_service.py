from common.enums.reduction_category import ReductionCategory
from core.domain.reduction_data import ReductionData
from core.models.reduction_parameter_model import ReductionParameterModel


class ReductionBaseService:
    category: ReductionCategory

    def __init__(self, category: ReductionCategory):
        self.category = category

    def reduce(self, parameters: ReductionParameterModel, data) -> list[ReductionData]:
        pass
