from flask import Blueprint, jsonify, request
from flask.views import MethodView
from injector import inject

from businesses.enzyme_business import EnzymeBusiness
from common.attributes.route import route
from common.enums.similarity_type import SimilarityType


class EnzymeController(MethodView):
    @inject
    def __init__(self, enzyme_business: EnzymeBusiness):
        self.enzyme_business = enzyme_business

    blue_print = Blueprint('enzyme', __name__)

    @route('enzymes/<drugbank_id>', methods=['GET'])
    def get_enzymes(self, drugbank_id: str):
        enzymes = self.enzyme_business.get_enzymes(drugbank_id)

        if enzymes:
            return jsonify({'data': enzymes, 'status': True})
        else:
            return jsonify({'message': "No drug found for the provided DrugBank ID!", 'status': False})

    @route('drugEnzymes', methods=['GET'])
    def get_drug_enzymes(self):
        start = int(request.args.get('start', 0))
        length = int(request.args.get('length', 10))
        draw = int(request.args.get('draw', 1))

        columns, enzymes, total_number = self.enzyme_business.get_drug_enzymes(start, length)

        if enzymes:
            return jsonify({'columns': columns, 'draw': draw, 'recordsTotal': total_number,
                            'recordsFiltered': total_number, 'data': enzymes, 'status': True})
        else:
            return jsonify({'message': "No enzyme found!", 'status': False})

    @route('enzymeSimilarity', methods=['GET'])
    def get_enzyme_similarity(self):
        start = int(request.args.get('start', 0))
        length = int(request.args.get('length', 10))
        draw = int(request.args.get('draw', 1))
        similarity_type_str = request.args.get('similarityType')

        columns, enzymes, total_number = self.enzyme_business.get_enzyme_similarity(SimilarityType[similarity_type_str],
                                                                                    start, length)

        if enzymes:
            return jsonify({'columns': columns, 'draw': draw, 'recordsTotal': total_number,
                            'recordsFiltered': total_number, 'data': enzymes, 'status': True})
        else:
            return jsonify({'columns': columns, 'draw': draw, 'recordsTotal': total_number,
                            'recordsFiltered': total_number, 'data': enzymes,
                            'message': "No enzyme found!", 'status': False})

    @route('calculateEnzymeSimilarity', methods=['POST'])
    def calculate_enzyme_similarity(self):
        similarity_type_str = request.args.get('similarityType')

        self.enzyme_business.generate_similarity(SimilarityType[similarity_type_str])

        return jsonify({'status': True})
