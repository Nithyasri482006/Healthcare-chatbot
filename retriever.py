from rag.embeddings import embedding_model
import numpy as np


def retrieve_context(question, vector_store, text_chunks, k=3):

    # Convert question into embedding
    question_embedding = embedding_model.encode([question])

    # Search FAISS
    distances, indices = vector_store.search(
        np.array(question_embedding).astype("float32"),
        k
    )

    # Get matching text chunks
    results = []

    for index in indices[0]:
        results.append(text_chunks[index])

    return "\n".join(results)