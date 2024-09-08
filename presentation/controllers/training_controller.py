from flask import Blueprint, jsonify, request
from flask.views import MethodView
from injector import inject

from businesses.training_business import TrainingBusiness
from common.attributes.route import route
from common.enums.train_models import TrainModel
from core.models.train_request_model import TrainRequestModel


class TrainingController(MethodView):
    @inject
    def __init__(self, training_business: TrainingBusiness):
        self.training_business = training_business

    blue_print = Blueprint('training', __name__)

    @route('fillTrainingModels', methods=['GET'])
    def get_reduction_categories(self):
        types = [{"name": train_model.name, "value": train_model.value} for train_model in TrainModel]
        return jsonify(types)

    @route('train', methods=['POST'])
    def train(self):
        data = request.get_json()

        train_request = TrainRequestModel.from_dict(data)

        try:
            self.training_business.train(train_request)

        except Exception as e:
            return jsonify({'status': False, "error": str(e)})

        return jsonify({'status': True})
