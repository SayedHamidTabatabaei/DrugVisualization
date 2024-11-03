from datetime import datetime

from flask import Blueprint, jsonify, request
from flask.views import MethodView
from injector import inject

from businesses.job_business import JobBusiness
from businesses.training_business import TrainingBusiness
from common.attributes.route import route
from common.enums.compare_plot_type import ComparePlotType
from common.enums.loss_functions import LossFunctions
from common.enums.scenarios import Scenarios
from common.enums.train_models import TrainModel
from core.view_models.train_request_view_model import TrainRequestViewModel
from infrastructure.repositories.training_repository import TrainingRepository
from infrastructure.repositories.training_scheduled_repository import TrainingScheduledRepository


class TrainingController(MethodView):
    @inject
    def __init__(self, training_business: TrainingBusiness, job_business: JobBusiness,
                 training_scheduled_repository: TrainingScheduledRepository, training_repository: TrainingRepository):
        self.training_business = training_business
        self.job_business = job_business
        self.training_repository = training_repository
        self.training_scheduled_repository = training_scheduled_repository

    blue_print = Blueprint('training', __name__)

    @route('fillScenarios', methods=['GET'])
    def get_scenarios(self):
        types = [{"name": scenario.name, "value": scenario.value} for scenario in Scenarios]
        return jsonify(types)

    @route('trainingSampleCounts', methods=['GET'])
    def get_training_sample_counts(self):
        training_sample_counts = self.training_repository.get_training_sample_counts()

        types = [{"name": tsc, "value": tsc} for tsc in training_sample_counts]
        return jsonify(types)

    @route('fillTrainingModels', methods=['GET'])
    def get_training_models(self):
        scenario_str = request.args.get('scenario')

        if not scenario_str:
            return jsonify({'status': False, 'data': ''})

        scenario = Scenarios[scenario_str]

        types = [{"name": train_model.name, "value": train_model.value} for train_model in TrainModel if train_model.scenario == scenario]
        return jsonify(types)

    @route('fillLossFunctions', methods=['GET'])
    def fill_loss_functions(self):

        train_model_str = request.args.get('trainModel')

        if not train_model_str:
            return jsonify({'status': False, 'data': ''})

        train_model = TrainModel[train_model_str]

        types = [{"name": loss_function.display_name, "value": loss_function.value} for loss_function in LossFunctions.valid(train_model)]
        return jsonify(types)

    @route('get_training_model_description', methods=['GET'])
    def get_training_model_description(self):

        train_model_str = request.args.get('trainModel')

        if not train_model_str:
            return jsonify({'status': False, 'data': ''})

        train_model = TrainModel[train_model_str]

        return jsonify({'status': True, 'data': train_model.description})

    @route('get_scenario_description', methods=['GET'])
    def get_scenario_description(self):

        scenario_str = request.args.get('scenario')

        if not scenario_str:
            return jsonify({'status': False, 'data': ''})

        scenario = Scenarios[scenario_str]

        return jsonify({'status': True, 'data': scenario.description})

    @route('get_loss_formula', methods=['GET'])
    def get_loss_formula(self):

        loss_function_value = int(request.args.get('selected_loss'))

        if not loss_function_value:
            return jsonify({'status': False, 'data': ''})

        loss_function = LossFunctions.from_value(loss_function_value)

        return jsonify({'status': True, 'data': loss_function.formula})

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

        scenario_str = request.args.get('scenario')

        scenario = None

        if scenario_str:
            scenario = Scenarios[scenario_str]

        date_string = request.args.get('date')

        if date_string:
            date = datetime.strptime(date_string, '%Y-%m-%d')
        else:
            date = None

        min_sample_count = int(request.args.get('min_sample_count') or "0")

        train_history, total_number = self.training_business.get_history(scenario, train_model, date, min_sample_count, start, length)

        # if train_history:
        return jsonify({'draw': draw, 'recordsTotal': total_number,
                        'recordsFiltered': total_number, 'data': train_history, 'status': True})
        # else:
        #     return jsonify({'draw': draw, 'recordsTotal': total_number,
        #                     'recordsFiltered': total_number, 'data': train_history,
        #                     'message': "No enzyme found!", 'status': False})

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
        train_id = request.args.get('trainHistoryId')

        train_history_details = self.training_business.get_history_details(int(train_id))

        if train_history_details:
            return jsonify({'data': train_history_details, 'status': True})
        else:
            return jsonify({'data': train_history_details, 'message': "No data found!", 'status': False})

    @route('get_history_conditions', methods=['GET'])
    def get_history_conditions(self):
        train_id = request.args.get('trainHistoryId')

        train_history_conditions = self.training_business.get_history_conditions(int(train_id))

        if train_history_conditions:
            return jsonify({'data': train_history_conditions, 'status': True})
        else:
            return jsonify({'data': train_history_conditions, 'message': "No data found!", 'status': False})

    @route('get_history_model_information', methods=['GET'])
    def get_history_model_information(self):
        train_id = request.args.get('trainHistoryId')

        train_history_model_information = self.training_business.get_history_model_information(int(train_id))

        if train_history_model_information:
            return jsonify({'data': train_history_model_information, 'status': True})
        else:
            return jsonify({'data': train_history_model_information, 'message': "No data found!", 'status': False})

    @route('get_history_data_reports', methods=['GET'])
    def get_history_data_reports(self):
        train_id = request.args.get('trainHistoryId')

        train_history_data_reports = self.training_business.get_history_data_reports(int(train_id))

        if train_history_data_reports:
            return jsonify({'data': train_history_data_reports, 'status': True})
        else:
            return jsonify({'data': train_history_data_reports, 'message': "No data found!", 'status': False})

    @route('get_history_fold_result_details', methods=['GET'])
    def get_history_fold_result_details(self):
        train_id = request.args.get('trainHistoryId')

        train_history_fold_result_details = self.training_business.get_history_fold_result_details(int(train_id))

        if train_history_fold_result_details:
            return jsonify({'data': train_history_fold_result_details, 'status': True})
        else:
            return jsonify({'data': train_history_fold_result_details, 'message': "No data found!", 'status': False})

    @route('get_history_plots', methods=['GET'])
    def get_history_plots(self):
        train_id = request.args.get('trainHistoryId')

        images = self.training_business.get_history_plots(int(train_id))

        if images:
            return jsonify({'data': images, 'status': True})
        else:
            return jsonify({'data': images, 'message': "No images found!", 'status': False})

    @route('get_comparing_plot', methods=['GET'])
    def get_comparing_plot(self):

        train_ids = request.args.get('trainHistoryIds')
        compare_plot_type = ComparePlotType.get_enum_from_string(request.args.get('ComparePlotType'))

        if not train_ids:
            return jsonify({'message': "No images found!", 'status': False})

        image = self.training_business.get_comparing_plot([int(train_id) for train_id in
                                                           train_ids.split(',')],
                                                          compare_plot_type)

        if image:
            return jsonify({'image': image, 'status': True})
        else:
            return jsonify({'image': image, 'message': "No images found!", 'status': False})

    @route('get_per_category_result_details', methods=['GET'])
    def get_per_category_result_details(self):

        train_ids = request.args.get('trainHistoryIds')
        compare_plot_type = ComparePlotType.get_enum_from_string(request.args.get('ComparePlotType'))

        if not train_ids:
            return jsonify({'message': "No images found!", 'status': False})

        columns, result_details = self.training_business.get_per_category_result_details([int(train_id) for train_id in train_ids.split(',')],
                                                                                         compare_plot_type)

        if result_details:
            return jsonify({'columns': columns, 'data': result_details, 'status': True})
        else:
            return jsonify({'message': "No enzyme found!", 'status': False})

    @route('get_train_data_report', methods=['GET'])
    def get_train_data_report(self):

        train_ids = request.args.get('trainHistoryIds')

        if not train_ids:
            return jsonify({'message': "No images found!", 'status': False})

        columns, data = self.training_business.get_data_report([int(train_id) for train_id in train_ids.split(',')], 'train_interaction_ids')

        if data:
            return jsonify({'columns': columns, 'data': data, 'status': True})
        else:
            return jsonify({'message': "No enzyme found!", 'status': False})

    @route('get_validation_data_report', methods=['GET'])
    def get_validation_data_report(self):

        train_ids = request.args.get('trainHistoryIds')

        if not train_ids:
            return jsonify({'message': "No images found!", 'status': False})

        columns, data = self.training_business.get_data_report([int(train_id) for train_id in train_ids.split(',')], 'val_interaction_ids')

        if data:
            return jsonify({'columns': columns, 'data': data, 'status': True})
        else:
            return jsonify({'message': "No enzyme found!", 'status': False})

    @route('get_test_data_report', methods=['GET'])
    def get_test_data_report(self):

        train_ids = request.args.get('trainHistoryIds')

        if not train_ids:
            return jsonify({'message': "No images found!", 'status': False})

        columns, data = self.training_business.get_data_report([int(train_id) for train_id in train_ids.split(',')], 'test_interaction_ids')

        if data:
            return jsonify({'columns': columns, 'data': data, 'status': True})
        else:
            return jsonify({'message': "No enzyme found!", 'status': False})

    @route('get_data_summary', methods=['GET'])
    def get_data_summary(self):

        train_ids = request.args.get('trainHistoryIds')

        if not train_ids:
            return jsonify({'message': "No images found!", 'status': False})

        columns, data = self.training_business.get_data_summary([int(train_id) for train_id in train_ids.split(',')])

        if data:
            return jsonify({'columns': columns, 'data': data, 'status': True})
        else:
            return jsonify({'message': "No enzyme found!", 'status': False})

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
