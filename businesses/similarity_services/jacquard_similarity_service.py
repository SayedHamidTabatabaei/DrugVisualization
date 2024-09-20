from decimal import Decimal

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm

from businesses.similarity_services.similarity_base_service import SimilarityBaseService
from scipy.spatial.distance import pdist, squareform

from common.enums.category import Category
from common.enums.similarity_type import SimilarityType
from core.domain.similarity import Similarity
from core.repository_models.drug_smiles_dto import DrugSmilesDTO


class JacquardSimilarityService(SimilarityBaseService):
    def calculate_similarity(self, codes, values, columns_description):

        features = super().create_specific_matrix_by_column_names(values, codes, columns_description[0])

        jacquard = self.generate_jacquard_similarity(features)

        return jacquard

    @staticmethod
    def generate_jacquard_similarity(feature_matrix):
        jacquard_distances = pdist(feature_matrix, metric='jaccard')

        jacquard_similarity_matrix = 1 - squareform(jacquard_distances)

        return jacquard_similarity_matrix

    def calculate_similes_similarity(self, all_drugs: list[DrugSmilesDTO]) -> list[Similarity]:

        for drug in all_drugs:
            drug.fingerprint = self.smiles_to_fingerprint(drug.smiles)

        similarities: list[Similarity] = []

        for drug_1 in tqdm(all_drugs, desc="Processing SMILES"):
            for drug_2 in all_drugs:
                similarity = self.jacquard_similarity(drug_1.fingerprint, drug_2.fingerprint)

                similarities.append(Similarity(similarity_type=SimilarityType.Jacquard,
                                               category=Category.Substructure,
                                               drug_1=drug_1.id,
                                               drug_2=drug_2.id,
                                               value=similarity))

        return similarities

    @staticmethod
    def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
        """Convert a SMILES string to a molecular fingerprint."""
        try:
            molecule = Chem.MolFromSmiles(smiles)
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=n_bits)
            return fingerprint
        except Exception as e:
            print(e)
            raise

    @staticmethod
    def jacquard_similarity(fp1, fp2) -> Decimal:
        return DataStructs.FingerprintSimilarity(fp1, fp2)
