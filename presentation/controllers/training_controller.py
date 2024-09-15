from flask import Blueprint, jsonify, request
from flask.views import MethodView
from injector import inject

from businesses.training_business import TrainingBusiness
from common.attributes.route import route
from common.enums.compare_plot_type import ComparePlotType
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
        #
        # try:
        self.training_business.train(train_request)

        # except Exception as e:
        #     return jsonify({'status': False, "error": str(e)})

        return jsonify({'status': True})

    @route('get_history', methods=['GET'])
    def get_history(self):
        start = int(request.args.get('start', 0))
        length = int(request.args.get('length', 10))
        draw = int(request.args.get('draw', 1))
        train_model_str = request.args.get('trainModel')

        train_model = None

        if train_model_str:
            train_model = TrainModel[train_model_str]

        train_history, total_number = self.training_business.get_history(train_model, start, length)

        if train_history:
            return jsonify({'draw': draw, 'recordsTotal': total_number,
                            'recordsFiltered': total_number, 'data': train_history, 'status': True})
        else:
            return jsonify({'draw': draw, 'recordsTotal': total_number,
                            'recordsFiltered': total_number, 'data': train_history,
                            'message': "No enzyme found!", 'status': False})

    @route('get_history_details', methods=['GET'])
    def get_history_details(self):
        train_result_id = request.args.get('trainHistoryId')

        train_history_details = self.training_business.get_history_details(int(train_result_id))

        if train_history_details:
            return jsonify({'data': train_history_details, 'status': True})
        else:
            return jsonify({'data': train_history_details,
                            'message': "No enzyme found!", 'status': False})

    @route('get_history_conditions', methods=['GET'])
    def get_history_conditions(self):
        train_result_id = request.args.get('trainHistoryId')

        train_history_conditions = self.training_business.get_history_conditions(int(train_result_id))

        if train_history_conditions:
            return jsonify({'data': train_history_conditions, 'status': True})
        else:
            return jsonify({'data': train_history_conditions,
                            'message': "No enzyme found!", 'status': False})

    @route('get_history_plots', methods=['GET'])
    def get_history_plots(self):
        train_result_id = request.args.get('trainHistoryId')

        images = self.training_business.get_history_plots(int(train_result_id))

        if images:
            return jsonify({'data': images, 'status': True})
        else:
            return jsonify({'data': images, 'message': "No images found!", 'status': False})

    @route('get_comparing_plots', methods=['GET'])
    def get_comparing_plots(self):

        train_result_ids = request.args.get('trainHistoryIds')

        images = self.training_business.get_comparing_plots([int(train_result_id) for train_result_id in
                                                             train_result_ids.split(',')])

        if images:
            return jsonify({'data': images, 'status': True})
        else:
            return jsonify({'data': images, 'message': "No images found!", 'status': False})

    @route('get_comparing_plot', methods=['GET'])
    def get_comparing_plot(self):

        train_result_ids = request.args.get('trainHistoryIds')
        compare_plot_type = ComparePlotType.get_enum_from_string(request.args.get('ComparePlotType'))

        image = self.training_business.get_comparing_plot([int(train_result_id) for train_result_id in
                                                           train_result_ids.split(',')],
                                                          compare_plot_type)

        if image:
            return jsonify({'image': image, 'status': True})
        else:
            return jsonify({'image': image, 'message': "No images found!", 'status': False})
