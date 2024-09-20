from businesses.trains.train_plan1 import TrainPlan1
from businesses.trains.train_plan2 import TrainPlan2
from businesses.trains.train_plan3 import TrainPlan3
from businesses.trains.train_plan4 import TrainPlan4
from businesses.trains.train_plan5 import TrainPlan5
from businesses.trains.train_plan6 import TrainPlan6
from businesses.trains.train_plan_base import TrainPlanBase
from common.enums.train_models import TrainModel


def get_instance(category: TrainModel) -> TrainPlanBase:
    if category == TrainModel.SimpleOneInput:
        return TrainPlan1(category)
    elif category == TrainModel.JoinSimplesBeforeSoftmax:
        return TrainPlan2(category)
    elif category == TrainModel.SumSoftmaxOutputs:
        return TrainPlan3(category)
    elif category == TrainModel.AutoEncoderWithDNN:
        return TrainPlan4(category)
    elif category == TrainModel.ContactDataWithOneDNN:
        return TrainPlan5(category)
    elif category == TrainModel.KNN:
        return TrainPlan6(category)
    else:
        raise ValueError("No suitable subclass found")
