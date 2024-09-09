from businesses.trains.train_plan2 import TrainPlan2
from businesses.trains.train_plan_base import TrainPlanBase
from common.enums.train_models import TrainModel


def get_instance(category: TrainModel) -> TrainPlanBase:
    match category:
        case TrainModel.SimpleOneInput:
            return TrainPlan1(category)
        case TrainModel.JoinSimplesBeforeSoftmax:
            return TrainPlan2(category)
        case _:
            raise ValueError("No suitable subclass found")