from dataclasses import dataclass

from common.enums.compare_plot_type import ComparePlotType


@dataclass
class ComparePlotDTO:
    compare_plot_type: ComparePlotType
    datas: []
    labels: []
