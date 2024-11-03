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
            print(f'We get 100 atoms of {len(mol.GetAtoms())}')
            break
        feature_matrix[i, 0] = atom.GetAtomicNum()
        feature_matrix[i, 1] = atom.GetDegree()  # Degree of the atom
        feature_matrix[i, 2] = atom.GetFormalCharge()
        feature_matrix[i, 3] = atom.GetHybridization().real  # Hybridization state
        feature_matrix[i, 4] = atom.GetIsAromatic()  # Aromaticity as binary

    return feature_matrix


def smiles_to_adjacency_matrix(smiles, max_atoms=100):
    # Convert SMILES to an RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # Initialize an empty adjacency matrix
    adjacency_matrix = np.zeros((max_atoms, max_atoms), dtype=int)

    # Populate the adjacency matrix based on bonds
    for bond in mol.GetBonds():
        # Get the indices of the atoms connected by the bond
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Only add bonds within the max_atoms limit
        if i < max_atoms and j < max_atoms:
            adjacency_matrix[i, j] = 1
            adjacency_matrix[j, i] = 1  # Symmetric for undirected graph

    return adjacency_matrix  # Return the actual molecule size matrix


def smiles_to_adjacency_matrix_atoms(smiles):
    # Convert SMILES to an RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # Get the number of atoms in the molecule
    num_atoms = mol.GetNumAtoms()

    # Get atom symbols for labeling the matrix
    atom_names = [atom.GetSymbol() for atom in mol.GetAtoms()]

    # Initialize an empty adjacency matrix
    adjacency_matrix = np.zeros((num_atoms, num_atoms), dtype=int)

    # Populate the adjacency matrix based on bonds
    for bond in mol.GetBonds():
        # Get the indices of the atoms connected by the bond
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Add bonds to the adjacency matrix
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1  # Symmetric for undirected graph

    return atom_names, adjacency_matrix
