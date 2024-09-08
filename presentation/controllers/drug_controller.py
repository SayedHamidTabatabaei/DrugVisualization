from flask import Blueprint, jsonify, request
from flask.views import MethodView
from injector import inject

from businesses.drug_business import DrugBusiness
from common.attributes.route import route
from common.enums.similarity_type import SimilarityType


class DrugController(MethodView):
    @inject
    def __init__(self, drug_business: DrugBusiness):
        self.drug_business = drug_business

    blue_print = Blueprint('drug', __name__)

    @route('drugList', methods=['GET'])
    def get_list(self):

        data = self.drug_business.get_list()

        if data:
            return jsonify({'data': data, 'status': True})
        else:
            return jsonify({'message': "There is an error to find drugs!", 'status': False})

    @route('visualization/<drugbank_id>', methods=['GET'])
    def get_visualization(self, drugbank_id: str):
        mol_block = self.drug_business.generate_visualization(drugbank_id)

        if mol_block:
            return jsonify({'mol_block': mol_block, 'status': True})
        else:
            return jsonify({'message': "No molecule block found for the provided DrugBank ID!", 'status': False})

    @route('information/<drugbank_id>', methods=['GET'])
    def get_information(self, drugbank_id: str):
        drug_information = self.drug_business.get_information(drugbank_id)

        if drug_information:
            return jsonify({'data': drug_information, 'status': True})
        else:
            return jsonify({'message': "No drug found for the provided DrugBank ID!", 'status': False})

    @route('interactions/<drugbank_id>', methods=['GET'])
    def get_interactions(self, drugbank_id: str):
        interactions = self.drug_business.get_interactions(drugbank_id)

        if interactions:
            return jsonify({'data': interactions, 'status': True})
        else:
            return jsonify({'message': "No drug found for the provided DrugBank ID!", 'status': False})

    @route('smilesSimilarity', methods=['GET'])
    def get_smiles_similarity(self):
        start = int(request.args.get('start', 0))
        length = int(request.args.get('length', 10))
        draw = int(request.args.get('draw', 1))
        similarity_type_str = request.args.get('similarityType')

        columns, smiles, total_number = self.drug_business.get_smiles_similarity(SimilarityType[similarity_type_str],
                                                                                 start, length)

        if smiles:
            return jsonify({'columns': columns, 'draw': draw, 'recordsTotal': total_number,
                            'recordsFiltered': total_number, 'data': smiles, 'status': True})
        else:
            return jsonify({'columns': columns, 'draw': draw, 'recordsTotal': total_number,
                            'recordsFiltered': total_number, 'data': smiles,
                            'message': "No smiles found!", 'status': False})

    @route('calculateSmilesSimilarity', methods=['POST'])
    def calculate_smiles_similarity(self):
        similarity_type_str = request.args.get('similarityType')

        self.drug_business.generate_similarity(SimilarityType[similarity_type_str])

        return jsonify({'status': True})
