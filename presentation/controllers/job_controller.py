from flask import Blueprint, jsonify
from flask.views import MethodView
from injector import inject

from businesses.job_business import JobBusiness
from common.attributes.route import route
from infrastructure.repositories.job_repository import JobRepository


class JobController(MethodView):
    @inject
    def __init__(self, job_business: JobBusiness, job_repository: JobRepository):
        self.job_repository = job_repository
        self.job_business = job_business

    blue_print = Blueprint('job', __name__)

    @route('incorrect_job_delete', methods=['DELETE'])
    def incorrect_job_delete(self):
        self.job_repository.incorrect_job_delete()

        return jsonify({"status": True})

    @route('start_job', methods=['POST'])
    def start_job(self):
        self.job_business.training_job()

        return jsonify({"status": True})
