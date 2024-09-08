from flask import Blueprint, jsonify, request
from flask.views import MethodView
from injector import inject

from businesses.target_business import TargetBusiness
from common.attributes.route import route
from common.enums.similarity_type import SimilarityType


class TargetController(MethodView):
    @inject
    def __init__(self, target_business: TargetBusiness):
        self.target_business = target_business

    blue_print = Blueprint('target', __name__)

    @route('targets/<drugbank_id>', methods=['GET'])
    def get_targets(self, drugbank_id: str):
        targets = self.target_business.get_targets(drugbank_id)

        if targets:
            return jsonify({'data': targets, 'status': True})
        else:
            return jsonify({'message': "No drug found for the provided DrugBank ID!", 'status': False})

    @route('drugTargets', methods=['GET'])
    def get_drug_targets(self):
        start = int(request.args.get('start', 0))
        length = int(request.args.get('length', 10))
        draw = int(request.args.get('draw', 1))

        columns, targets, total_number = self.target_business.get_drug_targets(start, length)

        if targets:
            return jsonify({'columns': columns, 'draw': draw, 'recordsTotal': total_number,
                            'recordsFiltered': total_number, 'data': targets, 'status': True})
        else:
            return jsonify({'message': "No targets found!", 'status': False})

    @route('targetSimilarity', methods=['GET'])
    def get_target_similarity(self):
        start = int(request.args.get('start', 0))
        length = int(request.args.get('length', 10))
        draw = int(request.args.get('draw', 1))
        similarity_type_str = request.args.get('similarityType')

        columns, targets, total_number = self.target_business.get_target_similarity(SimilarityType[similarity_type_str],
                                                                                    start, length)

        if targets:
            return jsonify({'columns': columns, 'draw': draw, 'recordsTotal': total_number,
                            'recordsFiltered': total_number, 'data': targets, 'status': True})
        else:
            return jsonify({'columns': columns, 'draw': draw, 'recordsTotal': total_number,
                            'recordsFiltered': total_number, 'data': targets,
                            'message': "No target found!", 'status': False})

    @route('calculateTargetSimilarity', methods=['POST'])
    def calculate_target_similarity(self):
        similarity_type_str = request.args.get('similarityType')

        self.target_business.generate_similarity(SimilarityType[similarity_type_str])

        return jsonify({'status': True})
