from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity


def generate_jacquard_similarity(feature_matrix):

    jacquard_distances = pdist(feature_matrix, metric='jaccard')

    jacquard_similarity_matrix = 1 - squareform(jacquard_distances)

    return jacquard_similarity_matrix


def generate_cosine_similarity(feature_matrix):

    cosine_sim = cosine_similarity(feature_matrix)

    return cosine_sim


def generate_column_names(base_matrix, drugbank_id_index=0, column_count=0):
    columns = []

    for i in range(column_count):
        columns.append('')

    for i in range(len(base_matrix[0])):
        columns.append(base_matrix[0][i][drugbank_id_index])

    return columns


def generate_combined_name_similarity(base_matrix, jacquard_matrix, first_enzyme_column_index=0):
    return [list(base[:first_enzyme_column_index]) + list(map(str, jacquard))
            for base, jacquard in zip(base_matrix, jacquard_matrix)]


def get_feature_matrix(base_matrix, first_column_index):
    feature_matrix = []
    for i in range(len(base_matrix)):

        row_features = []
        for j in range(first_column_index, len(base_matrix[i])):
            row_features.append(int(base_matrix[i][j]))

        feature_matrix.append(row_features)
    return feature_matrix
