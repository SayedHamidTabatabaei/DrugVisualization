from flask import Blueprint, jsonify, request
from flask.views import MethodView
from injector import inject

from businesses.pathway_business import PathwayBusiness
from common.attributes.route import route
from common.enums.similarity_type import SimilarityType


class PathwayController(MethodView):
    @inject
    def __init__(self, pathway_business: PathwayBusiness):
        self.pathway_business = pathway_business

    blue_print = Blueprint('pathway', __name__)

    @route('pathways/<drugbank_id>', methods=['GET'])
    def get_pathways(self, drugbank_id: str):
        pathways = self.pathway_business.get_pathways(drugbank_id)

        if pathways:
            return jsonify({'data': pathways, 'status': True})
        else:
            return jsonify({'message': "No drug found for the provided DrugBank ID!", 'status': False})

    @route('drugPathways', methods=['GET'])
    def get_drug_pathways(self):
        start = int(request.args.get('start', 0))
        length = int(request.args.get('length', 10))
        draw = int(request.args.get('draw', 1))

        columns, pathways, total_number = self.pathway_business.get_drug_pathways(start, length)

        if pathways:
            return jsonify({'columns': columns, 'draw': draw, 'recordsTotal': total_number,
                            'recordsFiltered': total_number, 'data': pathways, 'status': True})
        else:
            return jsonify({'message': "No pathway found!", 'status': False})

    @route('pathwaySimilarity', methods=['GET'])
    def get_pathway_similarity(self):
        start = int(request.args.get('start', 0))
        length = int(request.args.get('length', 10))
        draw = int(request.args.get('draw', 1))
        similarity_type_str = request.args.get('similarityType')

        columns, pathways, total_number = self.pathway_business.get_pathway_similarity(
            SimilarityType[similarity_type_str],
            start, length)

        if pathways:
            return jsonify({'columns': columns, 'draw': draw, 'recordsTotal': total_number,
                            'recordsFiltered': total_number, 'data': pathways, 'status': True})
        else:
            return jsonify({'columns': columns, 'draw': draw, 'recordsTotal': total_number,
                            'recordsFiltered': total_number, 'data': pathways,
                            'message': "No pathway found!", 'status': False})

    @route('calculatePathwaySimilarity', methods=['POST'])
    def calculate_pathway_similarity(self):
        similarity_type_str = request.args.get('similarityType')

        self.pathway_business.generate_similarity(SimilarityType[similarity_type_str])

        return jsonify({'status': True})
