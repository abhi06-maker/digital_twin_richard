"""
rag.py — Retrieval-Augmented Generation for Feynman Digital Twin.

Retrieves the most relevant chunks from ChromaDB given a user query,
so the agent can ground its answers in Feynman's actual words/ideas.
"""

import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "feynman_knowledge"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 4          # number of chunks to retrieve
MIN_RELEVANCE = 0.3  # cosine distance threshold (lower = more similar)


class FeynmanRAG:
    def __init__(self):
        self._collection = None
        self._ef = None

    def _load(self):
        """Lazy-load ChromaDB collection."""
        if self._collection is not None:
            return True

        if not Path(CHROMA_DIR).exists():
            return False

        try:
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBED_MODEL
            )
            self._collection = client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=self._ef
            )
            count = self._collection.count()
            if count == 0:
                self._collection = None
                return False
            return True
        except Exception as e:
            print(f"[RAG] Could not load collection: {e}")
            return False

    def retrieve(self, query: str, top_k: int = TOP_K) -> str:
        """
        Given a query, retrieve the top_k most relevant text chunks.
        Returns a formatted string ready for the system prompt.
        """
        if not self._load():
            return ""  # RAG not available, agent answers from base knowledge

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(top_k, self._collection.count()),
                include=["documents", "metadatas", "distances"]
            )

            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            # Filter by relevance
            relevant = [
                (doc, meta, dist)
                for doc, meta, dist in zip(docs, metas, distances)
                if dist < (1.0 - MIN_RELEVANCE)  # cosine distance: 0=identical, 2=opposite
            ]

            if not relevant:
                return ""

            parts = []
            for i, (doc, meta, dist) in enumerate(relevant, 1):
                source = meta.get("source", "unknown")
                relevance = round((1 - dist) * 100, 1)
                parts.append(
                    f"[Source: {source} | Relevance: {relevance}%]\n{doc.strip()}"
                )

            return "\n\n---\n\n".join(parts)

        except Exception as e:
            print(f"[RAG] Retrieval error: {e}")
            return ""

    def is_available(self) -> bool:
        return self._load()

    def doc_count(self) -> int:
        if not self._load():
            return 0
        return self._collection.count()


# Singleton for use across the app
rag = FeynmanRAG()