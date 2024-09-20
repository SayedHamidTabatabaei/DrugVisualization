from businesses.similarity_services.similarity_base_service import SimilarityBaseService


class OriginalSimilarityService(SimilarityBaseService):
    def calculate_similarity(self, codes, values, columns_description):

        features = super().create_specific_matrix_by_column_names(values, codes, columns_description[0])

        return features
