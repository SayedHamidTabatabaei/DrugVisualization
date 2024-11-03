from dataclasses import dataclass

from numpy import ndarray


@dataclass
class DrugInformationDTO:
    id: int
    drugbank_drug_id: int
    drug_name: str
    drugbank_id: str
    smiles: str
    drug_type: str
    description: str
    average_mass: int
    monoisotopic_mass: int
    state: str
    indication: str
    pharmacodynamics: str
    mechanism_of_action: str
    toxicity: str
    metabolism: str
    absorption: str
    half_life: str
    protein_binding: str
    route_of_elimination: str
    volume_of_distribution: str
    clearance: str
    classification_description: str
    classification_direct_parent: str
    classification_kingdom: str
    classification_superclass: str
    classification_class_category: str
    classification_subclass: str
    bioavailability: int
    ghose_filter: int
    h_bond_acceptor_count: int
    h_bond_donor_count: int
    log_p: float
    log_s: float
    mddr_like_rule: int
    molecular_formula: str
    molecular_weight: float
    monoisotopic_weight: float
    number_of_rings: int
    physiological_charge: int
    pka_strongest_acidic: float
    pka_strongest_basic: float
    polar_surface_area: float
    polarizability: float
    refractivity: float
    rotatable_bond_count: int
    rule_of_five: int
    water_solubility: str
    rdkit_3d: str
    rdkit_2d: str
    has_enzyme: bool
    has_pathway: bool
    has_target: bool
    has_smiles: bool
