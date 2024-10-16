from core.repository_models.interaction_embedding_dto import InteractionEmbeddingDTO
from core.repository_models.interaction_text_embedding_dto import InteractionTextEmbeddingDTO


def map_interaction_embedding(query_results) -> list[InteractionEmbeddingDTO]:
    embeddings = []
    for result in query_results:
        id, drug1_id, drugbank_id1, drug2_id, drugbank_id2, embedding_type, text_type, embedding = result
        embedding_entity = InteractionEmbeddingDTO(id=id,
                                                   drug1_id=drug1_id,
                                                   drugbank1_id=drugbank_id1,
                                                   drug2_id=drug2_id,
                                                   drugbank2_id=drugbank_id2,
                                                   embedding_type=embedding_type,
                                                   text_type=text_type,
                                                   embedding=embedding)
        embeddings.append(embedding_entity)

    return embeddings


def map_interaction_embedding_dict(query_results) -> dict:
    embeddings = {}
    for result in query_results:
        id, drug1_id, drugbank_id1, drug2_id, drugbank_id2, embedding_type, text_type, embedding = result
        embeddings[id] = embedding

    return embeddings


def map_interaction_text_embedding(query_results) -> list[InteractionTextEmbeddingDTO]:
    embeddings = []
    for result in query_results:
        id, drug1_id, drugbank_id1, drug1_name, drug2_id, drugbank_id2, drug2_name, embedding_type, text_type, embedding, text = result
        embedding_entity = InteractionTextEmbeddingDTO(id=id,
                                                       drug1_id=drug1_id,
                                                       drugbank1_id=drugbank_id1,
                                                       drug1_name=drug1_name,
                                                       drug2_id=drug2_id,
                                                       drugbank2_id=drugbank_id2,
                                                       drug2_name=drug2_name,
                                                       embedding_type=embedding_type,
                                                       text_type=text_type,
                                                       embedding=embedding,
                                                       text=text)
        embeddings.append(embedding_entity)

    return embeddings
