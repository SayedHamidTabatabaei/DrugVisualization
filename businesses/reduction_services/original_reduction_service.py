from businesses.reduction_services.reduction_base_service import ReductionBaseService
from common.enums.reduction_category import ReductionCategory
from core.domain.reduction_data import ReductionData
from core.models.reduction_parameter_model import ReductionParameterModel

reduction_category = ReductionCategory.OriginalData


class OriginalReductionService(ReductionBaseService):

    def reduce(self, parameters: ReductionParameterModel, data) -> list[ReductionData]:

        return [ReductionData(drug_id=key,
                              similarity_type=parameters.similarity_type,
                              category=parameters.category,
                              reduction_category=ReductionCategory.OriginalData,
                              reduction_values=str(values),
                              has_enzyme=True,
                              has_pathway=True,
                              has_target=True,
                              has_smiles=True) for key, values in data.items()]
