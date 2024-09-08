import numpy as np
import ast


def create_specific_matrix_by_column_names(base_matrix, column_codes, columns_description):
    columns = [desc[0] for desc in columns_description]

    filtered_columns = [col for col in columns if col in column_codes]

    matrix = []
    for row in base_matrix:
        filtered_row = [float(row[columns.index(col)]) for col in filtered_columns]
        matrix.append(filtered_row)

    return matrix


def create_specific_matrix_by_embedding_list(embedding_list: list[str]):
    matrix_strings = [matrix_str.replace(' ', ',') for matrix_str in embedding_list]

    matrices = []
    for matrix_str in matrix_strings:
        matrix_list = ast.literal_eval(matrix_str)

        matrix_array = np.array(matrix_list, dtype=np.float32)
        matrices.append(matrix_array.flatten())

    result = np.vstack(matrices)

    return result
