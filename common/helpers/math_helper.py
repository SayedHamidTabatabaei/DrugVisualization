import math


def round_up(value, decimal_places):
    factor = 10 ** decimal_places
    return math.ceil(value * factor) / factor