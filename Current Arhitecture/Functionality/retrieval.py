import numpy as np


# RETRIEVAL 
# retrieve top 20 chunks with the embedding model -> choose top 8 with the reranker model

def retrieve(embedding_model, index, chunks, reranker_model, query, k_initial=20, k_final=8):
    # Pasul A: Retrieval inițial (extrage mai multe fragmente)
    query_prefixed = "query: " + query
    q_emb = embedding_model.encode([query_prefixed])
    distances, indices = index.search(np.array(q_emb), k_initial)
    
    initial_chunks = [chunks[i] for i in indices[0]]

    if k_final == 0:  # 0 means skip reranking
        return initial_chunks


    # Pasul B: Reranking (Cross-Encoding)

    # Modelul compară perechi (întrebare, fragment) și dă un scor de relevanță
    sentence_combinations = [[query, chunk] for chunk in initial_chunks]
    rerank_scores = reranker_model.predict(sentence_combinations)
    
    # Pasul C: Sortăm fragmentele în funcție de scorul Reranker-ului
    reranked_results = sorted(
        list(zip(initial_chunks, rerank_scores)), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Returnăm doar primele k_final cele mai bune
    return [res[0] for res in reranked_results[:k_final]]