from businesses.reduction_services.autoencoder_reduction_service import AutoEncoderReductionService
from businesses.reduction_services.original_reduction_service import OriginalReductionService
from businesses.reduction_services.reduction_base_service import ReductionBaseService
from common.enums.reduction_category import ReductionCategory


def get_instance(category: ReductionCategory) -> ReductionBaseService:
    match category:
        case ReductionCategory.OriginalData:
            return OriginalReductionService(category)
        case ReductionCategory.AutoEncoder_Max:
            return AutoEncoderReductionService(category, 936)
        case ReductionCategory.AutoEncoder_Min:
            return AutoEncoderReductionService(category, 768)
        case ReductionCategory.AutoEncoder_Mean:
            return AutoEncoderReductionService(category, 852)
        case _:
            raise ValueError("No suitable subclass found")
