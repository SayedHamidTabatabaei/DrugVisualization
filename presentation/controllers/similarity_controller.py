from flask import Blueprint, jsonify, request
from flask.views import MethodView
from injector import inject

from businesses.similarity_business import SimilarityBusiness
from common.attributes.route import route
from common.enums.category import Category
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

    @route('filterUsageSimilarityTypes', methods=['GET'])
    def filter_usage_similarity_types(self):

        category = Category[request.args.get('category', 1)]

        types = [{"name": similarity_type.name, "value": similarity_type.value}
                 for similarity_type in self.similarity_business.get_similarities_by_category(category)]
        return jsonify(types)
