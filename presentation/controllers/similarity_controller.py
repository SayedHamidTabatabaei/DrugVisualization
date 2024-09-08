from flask import Blueprint, jsonify
from flask.views import MethodView
from injector import inject

from businesses.similarity_business import SimilarityBusiness
from common.attributes.route import route
from common.enums.similarity_type import SimilarityType


class SimilarityController(MethodView):
    @inject
    def __init__(self, similarity_business: SimilarityBusiness):
        self.similarity_business = similarity_business

    blue_print = Blueprint('similarity', __name__)

    @route('fillSimilarityTypes', methods=['GET'])
    def get_similarity_types(self):
        types = [{"name": similarity_type.name, "value": similarity_type.value} for similarity_type in SimilarityType]
        return jsonify(types)
