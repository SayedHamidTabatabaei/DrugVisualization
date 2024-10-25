import numpy as np
from rdkit import Chem


def smiles_to_feature_matrix(smiles, max_atoms=100):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    num_features = 5
    feature_matrix = np.zeros((max_atoms, num_features))

    for i, atom in enumerate(mol.GetAtoms()):
        if i >= max_atoms:
            break
        feature_matrix[i, 0] = atom.GetAtomicNum()
        feature_matrix[i, 1] = atom.GetDegree()  # Degree of the atom
        feature_matrix[i, 2] = atom.GetFormalCharge()
        feature_matrix[i, 3] = atom.GetHybridization().real  # Hybridization state
        feature_matrix[i, 4] = atom.GetIsAromatic()  # Aromaticity as binary

    return feature_matrix
