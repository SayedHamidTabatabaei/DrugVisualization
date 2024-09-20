from businesses.similarity_services.similarity_base_service import SimilarityBaseService
from sklearn.metrics.pairwise import cosine_similarity


class CosineSimilarityService(SimilarityBaseService):
    def calculate_similarity(self, codes, values, columns_description):

        features = super().create_specific_matrix_by_column_names(values, codes, columns_description[0])

        cosine = self.generate_cosine_similarity(features)

        return cosine

    @staticmethod
    def generate_cosine_similarity(feature_matrix):
        cosine_sim = cosine_similarity(feature_matrix)

        return cosine_sim
