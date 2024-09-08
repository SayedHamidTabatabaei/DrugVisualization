from flask import Blueprint, jsonify, request
from flask.views import MethodView
from injector import inject

from businesses.drug_embedding_business import DrugEmbeddingBusiness
from common.attributes.route import route
from common.enums.embedding_type import EmbeddingType
from common.enums.text_type import TextType


class DrugEmbeddingController(MethodView):
    @inject
    def __init__(self, drug_embedding_business: DrugEmbeddingBusiness):
        self.drug_embedding_business = drug_embedding_business

    blue_print = Blueprint('drugembedding', __name__)

    @route('fillPropertyNames', methods=['GET'])
    def get_text_types(self):
        types = [{"name": text_type.name, "value": text_type.name} for text_type in TextType]
        return jsonify(types)

    @route('fillEmbeddingCategories', methods=['GET'])
    def get_embedding_types(self):
        types = [{"id": embedding_type.value, "value": embedding_type.name, "label": embedding_type.name,
                  "name": embedding_type.name} for embedding_type in EmbeddingType]
        return jsonify(types)

    @route('textEmbedding', methods=['GET'])
    def get_text_embeddings(self):
        start = int(request.args.get('start', 0))
        length = int(request.args.get('length', 10))
        draw = int(request.args.get('draw', 1))
        text_type = request.args.get('text_type')
        embedding_type = request.args.get('embedding_type')

        embeddings, total_number = self.drug_embedding_business.get_all_embeddings(
            EmbeddingType[embedding_type], TextType[text_type], start, length)

        if embeddings:
            return jsonify({'draw': draw, 'recordsTotal': total_number,
                            'recordsFiltered': total_number, 'data': embeddings, 'status': True})
        else:
            return jsonify({'message': "No pathway found!", 'status': False})

    @route('calculateEmbedding', methods=['POST'])
    def calculate_embedding(self):
        text_type = request.args.get('text_type')
        embedding_type = request.args.get('embedding_type')

        self.drug_embedding_business.calculate_embeddings(EmbeddingType[embedding_type], TextType[text_type])

        return jsonify({'status': True})
