from flask import Blueprint, jsonify
from flask.views import MethodView
from injector import inject

from businesses.similarity_business import SimilarityBusiness
from common.attributes.route import route
from common.enums.similarity_type import SimilarityType
from infrastructure.repositories.job_repository import JobRepository


class JobController(MethodView):
    @inject
    def __init__(self, job_repository: JobRepository):
        self.job_repository = job_repository

    blue_print = Blueprint('job', __name__)

    @route('incorrect_job_delete', methods=['DELETE'])
    def get_similarity_types(self):
        self.job_repository.incorrect_job_delete()

        return jsonify({"status": True})
