import faiss


class FAISSRetriever:
    def __init__(self, embeddings, metadata):
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.metadata = metadata

    def search(self, query_embedding, top_k=5, filter_type=None):
        results = self.search_with_scores(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_type=filter_type
        )
        return [idx for idx, _ in results]

    def search_with_scores(self, query_embedding, top_k=5, filter_type=None):
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            meta = self.metadata[idx]

            if filter_type and meta.get("type") != filter_type:
                continue

            results.append((int(idx), float(dist)))

        return results
