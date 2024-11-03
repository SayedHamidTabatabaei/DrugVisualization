from dataclasses import dataclass, field
from typing import List


@dataclass
class FeatureDataDTO:
    drug_id: int
    drugbank_id: str
    drug_name: str
    drug_type: str
    features: List[float] = field(default_factory=list)
