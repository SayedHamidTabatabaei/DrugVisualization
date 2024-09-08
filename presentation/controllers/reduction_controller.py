from flask import Blueprint, jsonify, request
from flask.views import MethodView
from injector import inject

from businesses.reduction_business import ReductionBusiness
from common.attributes.route import route
from common.enums.category import Category
from common.enums.embedding_type import EmbeddingType
from common.enums.reduction_category import ReductionCategory
from common.enums.similarity_type import SimilarityType
from common.enums.text_type import TextType


class ReductionController(MethodView):
    @inject
    def __init__(self, reduction_business: ReductionBusiness):
        self.reduction_business = reduction_business

    blue_print = Blueprint('reduction', __name__)

    @route('fillReductionCategories', methods=['GET'])
    def get_reduction_categories(self):
        types = [{"id": reduction_category.value, "value": reduction_category.name, "label": reduction_category.name,
                  "name": reduction_category.name}
                 for reduction_category in ReductionCategory]
        return jsonify(types)

    @route('embedding', methods=['GET'])
    def get_reduction_embeddings(self):
        start = int(request.args.get('start', 0))
        length = int(request.args.get('length', 10))
        draw = int(request.args.get('draw', 1))
        reduction_category = request.args.get('reduction_category')
        text_type = request.args.get('text_type')
        embedding_type = request.args.get('embedding_type')

        embeddings, total_number = self.reduction_business.get_reduction_embeddings(
            ReductionCategory[reduction_category], EmbeddingType[embedding_type], TextType[text_type], start, length)

        if embeddings:
            return jsonify({'draw': draw, 'recordsTotal': total_number,
                            'recordsFiltered': total_number, 'data': embeddings, 'status': True})
        else:
            return jsonify({'message': "No pathway found!", 'status': False})

    @route('similarity', methods=['GET'])
    def get_reduction_similarities(self):
        start = int(request.args.get('start', 0))
        length = int(request.args.get('length', 10))
        draw = int(request.args.get('draw', 1))
        reduction_category = request.args.get('reduction_category')
        category = request.args.get('category')
        similarity_type = request.args.get('similarity_type')

        embeddings, total_number = self.reduction_business.get_reduction_similarity(
            ReductionCategory[reduction_category], SimilarityType[similarity_type], Category[category], start, length)

        if embeddings:
            return jsonify({'draw': draw, 'recordsTotal': total_number,
                            'recordsFiltered': total_number, 'data': embeddings, 'status': True})
        else:
            return jsonify({'message': "No pathway found!", 'status': False})

    @route('calculateSimilarity', methods=['POST'])
    def calculate_reduction_similarities(self):
        reduction_category = request.args.get('reduction_category')
        category = request.args.get('category')
        similarity_type = request.args.get('similarity_type')

        self.reduction_business.calculate_reduction_similarity(
            ReductionCategory[reduction_category], SimilarityType[similarity_type], Category[category])

        return jsonify({'status': True})

    @route('calculateEmbedding', methods=['POST'])
    def calculate_reduction_embeddings(self):
        text_type = request.args.get('text_type')
        embedding_type = request.args.get('embedding_type')
        reduction_category = request.args.get('reduction_category')

        self.reduction_business.calculate_reduction_embedding(
            ReductionCategory[reduction_category], TextType[text_type], EmbeddingType[embedding_type])

        return jsonify({'status': True})
