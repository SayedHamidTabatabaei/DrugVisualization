from core.repository_models.drug_embedding_dto import DrugEmbeddingDTO
from core.repository_models.text_embedding_dto import TextEmbeddingDTO


def map_drug_embedding(query_results) -> list[DrugEmbeddingDTO]:
    embeddings = []
    for result in query_results:
        id, drug_id, drugbank_id, embedding_type, text_type, embedding = result
        embedding_entity = DrugEmbeddingDTO(id=id,
                                            drug_id=drug_id,
                                            drugbank_id=drugbank_id,
                                            embedding_type=embedding_type,
                                            text_type=text_type,
                                            embedding=embedding)
        embeddings.append(embedding_entity)

    return embeddings


def map_drug_embedding_dict(query_results) -> dict:
    embeddings = {}
    for result in query_results:
        id, drug_id, drugbank_id, embedding_type, text_type, embedding = result
        embeddings[drug_id] = embedding

    return embeddings


def map_text_embedding(query_results) -> list[TextEmbeddingDTO]:
    embeddings = []
    for result in query_results:
        id, drug_id, drugbank_id, drug_name, embedding_type, text_type, embedding, text = result
        embedding_entity = TextEmbeddingDTO(id=id,
                                            drug_id=drug_id,
                                            drugbank_id=drugbank_id,
                                            drug_name=drug_name,
                                            embedding_type=embedding_type,
                                            text_type=text_type,
                                            embedding=embedding,
                                            text=text)
        embeddings.append(embedding_entity)

    return embeddings
