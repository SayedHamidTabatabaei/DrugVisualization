from core.repository_models.feature_data_dto import FeatureDataDTO


def map_target_features(query_results) -> list[FeatureDataDTO]:
    target_feature = []
    for result in query_results:
        drug_id, drugbank_id, drug_name, drug_type, *feature_columns = result

        feature_data = FeatureDataDTO(
            drug_id=drug_id,
            drugbank_id=drugbank_id,
            drug_name=drug_name,
            drug_type=drug_type,
            features=[float(f) for f in feature_columns]
        )

        target_feature.append(feature_data)

    return target_feature
