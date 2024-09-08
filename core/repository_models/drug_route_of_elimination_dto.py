from dataclasses import dataclass


@dataclass
class DrugRouteOfEliminationDTO:
    id: int
    route_of_elimination: str
