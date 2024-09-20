from injector import inject

from businesses.base_business import BaseBusiness
from businesses.similarity_business import SimilarityBusiness
from common.enums.category import Category
from common.enums.similarity_type import SimilarityType
from infrastructure.repositories.drug_repository import DrugRepository
from infrastructure.repositories.enzyme_repository import EnzymeRepository
from infrastructure.repositories.pathway_repository import PathwayRepository
from infrastructure.repositories.target_repository import TargetRepository


class DrugBusiness(BaseBusiness):
    @inject
    def __init__(self, drug_repository: DrugRepository, enzyme_repository: EnzymeRepository,
                 target_repository: TargetRepository, pathway_repository: PathwayRepository,
                 similarity_business: SimilarityBusiness):
        BaseBusiness.__init__(self)
        self.drug_repository = drug_repository
        self.enzyme_repository = enzyme_repository
        self.target_repository = target_repository
        self.pathway_repository = pathway_repository
        self.similarity_business = similarity_business

    def get_information(self, drugbank_id):
        drug_information = self.drug_repository.get_drug_information_by_drugbank_id(drugbank_id)

        columns = ['id', 'drugbank_drug_id', 'drug_name', 'drugbank_id', 'smiles', 'drug_type', 'description',
                   'average_mass', 'monoisotopic_mass', 'state', 'indication', 'pharmacodynamics',
                   'mechanism_of_action', 'toxicity', 'metabolism', 'absorption', 'half_life', 'protein_binding',
                   'route_of_elimination', 'volume_of_distribution', 'clearance', 'classification_description',
                   'classification_direct_parent', 'classification_kingdom', 'classification_superclass',
                   'classification_class_category', 'classification_subclass', 'bioavailability', 'ghose_filter',
                   'h_bond_acceptor_count', 'h_bond_donor_count', 'log_p', 'log_s', 'mddr_like_rule',
                   'molecular_formula', 'molecular_weight', 'monoisotopic_weight', 'number_of_rings',
                   'physiological_charge', 'pka_strongest_acidic', 'pka_strongest_basic', 'polar_surface_area',
                   'polarizability', 'refractivity', 'rotatable_bond_count', 'rule_of_five', 'water_solubility',
                   'has_enzyme', 'has_pathway', 'has_target', 'rdkit_3d', 'rdkit_2d']

        data = [dict(zip(columns, row)) for row in drug_information[0]]

        return data

    def get_enzymes(self, drugbank_id):
        drug_id = self.drug_repository.get_id_by_drugbank_id(drugbank_id)

        enzymes = self.enzyme_repository.get_enzymes_by_drug_id(drug_id)

        columns = ['id', 'enzyme_code', 'enzyme_name', 'position', 'organism']

        data = [dict(zip(columns, row)) for row in enzymes[0]]

        return data

    def get_targets(self, drugbank_id):
        drug_id = self.drug_repository.get_id_by_drugbank_id(drugbank_id)

        targets = self.target_repository.get_targets_by_drug_id(drug_id)

        columns = ['id', 'target_code', 'target_name', 'position', 'organism']

        data = [dict(zip(columns, row)) for row in targets[0]]

        return data

    def get_pathways(self, drugbank_id):
        drug_id = self.drug_repository.get_id_by_drugbank_id(drugbank_id)

        pathways = self.pathway_repository.get_pathways_by_drug_id(drug_id)

        columns = ['id', 'pathway_code', 'pathway_name']

        data = [dict(zip(columns, row)) for row in pathways[0]]

        return data

    def get_interactions(self, drugbank_id):
        drug_id = self.drug_repository.get_id_by_drugbank_id(drugbank_id)

        interactions = self.drug_repository.get_interactions_by_drug_id(drug_id)

        columns = ['destination_drugbank_id', 'destination_drug_name', 'description']

        data = [dict(zip(columns, row)) for row in interactions[0]]

        return data

    def get_active_drug_number(self):

        drug_count = self.drug_repository.get_active_drug_number()

        return drug_count

    def get_list(self):

        drugs = self.drug_repository.get_list()

        columns = ['id', 'drug_name', 'drugbank_id', 'state']

        data = [dict(zip(columns, row)) for row in drugs[0]]

        return data

    def generate_visualization(self, drugbank_id, is_3d=True):
        if is_3d:
            return self.generate_3d_visualization(drugbank_id)
        else:
            return self.generate_2d_visualization(drugbank_id)

    def generate_3d_visualization(self, drugbank_id):
        rdkit = self.drug_repository.find_rdkit_by_drugbankid(drugbank_id)

        if rdkit and len(rdkit) > 0 and len(rdkit[0]) > 0 and len(rdkit[0][0]) > 1:
            return rdkit[0][0][1]
        else:
            return None

    def generate_2d_visualization(self, drugbank_id):
        rdkit = self.drug_repository.find_rdkit_by_drugbankid(drugbank_id)

        if rdkit and len(rdkit) > 0 and len(rdkit[0]) > 0 and len(rdkit[0][0]) > 1:
            return rdkit[0][0][0]
        else:
            return None

    def get_smiles_similarity(self, similarity_type: SimilarityType, start: int, length: int):
        columns, data, total_number = self.similarity_business.get_similarity_grid_data(
            similarity_type, Category.Substructure, True, True, True, True, start, length)

        return columns, data, total_number

    def generate_similarity(self, similarity_type: SimilarityType):
        all_drug_smiles = self.drug_repository.get_all_drug_smiles()

        self.similarity_business.calculate_smiles_similarity(similarity_type, all_drug_smiles)
