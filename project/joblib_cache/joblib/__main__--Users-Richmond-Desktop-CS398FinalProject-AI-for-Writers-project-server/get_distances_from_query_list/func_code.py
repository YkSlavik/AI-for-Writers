# first line: 87
@memory.cache
def get_distances_from_query_list(query_list, texts):
    list_emb = [get_embedding(text) for text in texts]
    distances = []
    for query in query_list:
        query_emb = get_embedding(query)
        distances.append(distances_from_embeddings(query_emb, list_emb))
    return distances
