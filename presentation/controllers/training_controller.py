from flask import Blueprint, jsonify, request
from flask.views import MethodView
from injector import inject

from businesses.job_business import JobBusiness
from businesses.training_business import TrainingBusiness
from common.attributes.route import route
from common.enums.compare_plot_type import ComparePlotType
from common.enums.train_models import TrainModel
from core.view_models.train_request_view_model import TrainRequestViewModel
from infrastructure.repositories.training_scheduled_repository import TrainingScheduledRepository


class TrainingController(MethodView):
    @inject
    def __init__(self, training_business: TrainingBusiness, job_business: JobBusiness,
                 training_scheduled_repository: TrainingScheduledRepository):
        self.training_business = training_business
        self.job_business = job_business
        self.training_scheduled_repository = training_scheduled_repository

    blue_print = Blueprint('training', __name__)

    @route('fillTrainingModels', methods=['GET'])
    def get_training_models(self):
        types = [{"name": train_model.name, "value": train_model.value} for train_model in TrainModel]
        return jsonify(types)

    @route('get_training_model_description', methods=['GET'])
    def get_training_model_description(self):

        train_model_str = request.args.get('trainModel')

        if not train_model_str:
            return jsonify({'status': False, 'data': ''})

        train_model = TrainModel[train_model_str]

        return jsonify({'status': True, 'data': train_model.description})

    @route('train', methods=['POST'])
    def train(self):
        data = request.get_json()

        train_request = TrainRequestViewModel.from_dict(data)

        self.training_business.schedule_train(train_request)

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

    @route('get_schedules', methods=['GET'])
    def get_schedules(self):
        train_model_str = request.args.get('trainModel')

        train_model = None

        if train_model_str:
            train_model = TrainModel[train_model_str]

        train_schedules = self.training_business.get_training_scheduled(train_model)

        return jsonify({'data': train_schedules, 'status': True})

    @route('get_history_details', methods=['GET'])
    def get_history_details(self):
        train_result_id = request.args.get('trainHistoryId')

        train_history_details = self.training_business.get_history_details(int(train_result_id))

        if train_history_details:
            return jsonify({'data': train_history_details, 'status': True})
        else:
            return jsonify({'data': train_history_details,
                            'message': "No data found!", 'status': False})

    @route('get_history_conditions', methods=['GET'])
    def get_history_conditions(self):
        train_result_id = request.args.get('trainHistoryId')

        train_history_conditions = self.training_business.get_history_conditions(int(train_result_id))

        if train_history_conditions:
            return jsonify({'data': train_history_conditions, 'status': True})
        else:
            return jsonify({'data': train_history_conditions,
                            'message': "No data found!", 'status': False})

    @route('get_history_plots', methods=['GET'])
    def get_history_plots(self):
        train_result_id = request.args.get('trainHistoryId')

        images = self.training_business.get_history_plots(int(train_result_id))

        if images:
            return jsonify({'data': images, 'status': True})
        else:
            return jsonify({'data': images, 'message': "No images found!", 'status': False})

    # @route('get_comparing_plots', methods=['GET'])
    # def get_comparing_plots(self):
    #
    #     train_result_ids = request.args.get('trainHistoryIds')
    #
    #     images = self.training_business.get_comparing_plots([int(train_result_id) for train_result_id in
    #                                                          train_result_ids.split(',')])
    #
    #     if images:
    #         return jsonify({'data': images, 'status': True})
    #     else:
    #         return jsonify({'data': images, 'message': "No images found!", 'status': False})

    @route('get_comparing_plot', methods=['GET'])
    def get_comparing_plot(self):

        train_result_ids = request.args.get('trainHistoryIds')
        compare_plot_type = ComparePlotType.get_enum_from_string(request.args.get('ComparePlotType'))

        if not train_result_ids:
            return jsonify({'message': "No images found!", 'status': False})

        image = self.training_business.get_comparing_plot([int(train_result_id) for train_result_id in
                                                           train_result_ids.split(',')],
                                                          compare_plot_type)

        if image:
            return jsonify({'image': image, 'status': True})
        else:
            return jsonify({'image': image, 'message': "No images found!", 'status': False})

    @route('training_schedule_delete', methods=['DELETE'])
    def training_schedule_delete(self):

        id = int(request.args.get('id'))
        row_count = self.training_scheduled_repository.delete(id)

        if row_count and row_count != 0:
            return jsonify({'status': True})
        else:
            return jsonify({'message': "No data found!", 'status': False})

    @route('run_train', methods=['POST'])
    def run_train(self):

        id = int(request.args.get('id'))
        row_count = self.job_business.training_job(id)

        if row_count and row_count != 0:
            return jsonify({'status': True})
        else:
            return jsonify({'message': "No data found!", 'status': False})
