from enum import Enum


class SimilarityType(Enum):
    Original = 0
    Jacquard = 1
    Cosine = 2
    Euclidean = 3
    Manhattan = 4
    Hamming = 5
    Pearson = 6
    Spearman = 7
    Mahalanobis = 8
    Dice = 9
    Tanimoto = 10
    Kullback = 11
    Bhattacharyya = 12
    Edit = 13
    DynamicTime = 14
    Hausdorff = 15,
    DeepDDISmiles = 20
