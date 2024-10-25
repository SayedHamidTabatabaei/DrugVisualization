# This Similarity service just used for SMILES codes
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm

from businesses.similarity_services.similarity_base_service import SimilarityBaseService
from common.enums.category import Category
from common.enums.similarity_type import SimilarityType
from core.domain.similarity import Similarity
from core.repository_models.drug_smiles_dto import DrugSmilesDTO


class DrugStructureSimilarityService(SimilarityBaseService):
    def calculate_similes_similarity(self, all_drugs: list[DrugSmilesDTO]) -> list[Similarity]:

        result: list[Similarity] = []

        for drug in all_drugs:
            try:
                each_smiles = drug.smiles
                drug_mol = Chem.MolFromSmiles(each_smiles)
                drug.fingerprint = AllChem.GetMorganFingerprintAsBitVect(drug_mol, 2, nBits=2048)

            except Exception as e:
                print(f"{drug.id} got error {e}")
                continue

        for drug_1 in tqdm(all_drugs, "Calculating Similarity ..."):
            if not drug_1.fingerprint:
                continue

            for drug_2 in all_drugs:
                if not drug_2.fingerprint:
                    continue

                score = DataStructs.DiceSimilarity(drug_1.fingerprint, drug_2.fingerprint)

                result.append(Similarity(similarity_type=SimilarityType.DeepDDISmiles,
                                         category=Category.Substructure,
                                         drug_1=drug_1.id,
                                         drug_2=drug_2.id,
                                         value=score))

        return result
