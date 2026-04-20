"""
Core RAG Pipeline
A minimal but realistic Retrieval-Augmented Generation pipeline
that can be poisoned by adversarial documents.

Architecture:
  Documents → Chunker → Embedder → Vector Store → Retriever → LLM → Response
"""

import os
import json
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Document:
    """A document in the RAG knowledge base."""
    doc_id: str
    content: str
    source: str
    metadata: Dict = field(default_factory=dict)
    is_poisoned: bool = False
    poisoning_type: Optional[str] = None

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:12]


@dataclass
class RetrievedContext:
    """Context retrieved from the vector store for a query."""
    query: str
    documents: List[Document]
    scores: List[float]

    @property
    def contains_poisoned(self) -> bool:
        return any(d.is_poisoned for d in self.documents)

    @property
    def poisoned_count(self) -> int:
        return sum(1 for d in self.documents if d.is_poisoned)


@dataclass
class RAGResponse:
    """Final response from the RAG pipeline."""
    query: str
    response: str
    context: RetrievedContext
    model: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    was_poisoned: bool = False
    poison_influence: Optional[str] = None


class SimpleVectorStore:
    """
    Lightweight in-memory vector store using TF-IDF-style similarity.
    No GPU or external dependencies required — designed for demo/lab use.
    For production, replace with FAISS, Chroma, Pinecone, or Weaviate.
    """

    def __init__(self):
        self.documents: List[Document] = []
        self._index: Dict[str, List[float]] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer."""
        import re
        return re.findall(r'\b\w+\b', text.lower())

    def _vectorize(self, tokens: List[str], vocab: List[str]) -> List[float]:
        """Create a simple term-frequency vector."""
        from collections import Counter
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        return [counts.get(word, 0) / total for word in vocab]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _build_index(self):
        """Build vocabulary and vectorize all documents."""
        all_tokens = []
        for doc in self.documents:
            all_tokens.extend(self._tokenize(doc.content))
        vocab = list(set(all_tokens))

        self._vocab = vocab
        self._index = {}
        for doc in self.documents:
            tokens = self._tokenize(doc.content)
            self._index[doc.doc_id] = self._vectorize(tokens, vocab)

    def add_documents(self, documents: List[Document]):
        self.documents.extend(documents)
        self._build_index()

    def add_document(self, document: Document):
        self.documents.append(document)
        self._build_index()

    def search(self, query: str, top_k: int = 3) -> Tuple[List[Document], List[float]]:
        if not self.documents:
            return [], []

        query_tokens = self._tokenize(query)
        query_vec = self._vectorize(query_tokens, self._vocab)

        scores = []
        for doc in self.documents:
            doc_vec = self._index.get(doc.doc_id, [0] * len(self._vocab))
            score = self._cosine_similarity(query_vec, doc_vec)
            scores.append((doc, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:top_k]
        return [item[0] for item in top], [item[1] for item in top]

    def clear(self):
        self.documents = []
        self._index = {}

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def poisoned_count(self) -> int:
        return sum(1 for d in self.documents if d.is_poisoned)


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.
    Uses an LLM adapter for generation, vector store for retrieval.
    """

    def __init__(self, llm_adapter=None, top_k: int = 3):
        self.vector_store = SimpleVectorStore()
        self.llm_adapter = llm_adapter
        self.top_k = top_k
        self.query_log: List[RAGResponse] = []

    def ingest(self, documents: List[Document]):
        """Add documents to the knowledge base."""
        self.vector_store.add_documents(documents)
        print(f"[RAG] Ingested {len(documents)} documents. "
              f"Total: {self.vector_store.document_count} "
              f"({self.vector_store.poisoned_count} poisoned)")

    def inject_poison(self, document: Document):
        """Inject a single poisoned document into the knowledge base."""
        document.is_poisoned = True
        self.vector_store.add_document(document)
        print(f"[RAG] ⚠️  Poisoned document injected: [{document.poisoning_type}] {document.doc_id}")

    def retrieve(self, query: str) -> RetrievedContext:
        """Retrieve the most relevant documents for a query."""
        docs, scores = self.vector_store.search(query, top_k=self.top_k)
        return RetrievedContext(query=query, documents=docs, scores=scores)

    def _build_prompt(self, query: str, context: RetrievedContext) -> str:
        """Build the prompt sent to the LLM with retrieved context."""
        context_text = "\n\n---\n\n".join([
            f"[Source: {doc.source}]\n{doc.content}"
            for doc in context.documents
        ])
        return (
            f"You are a helpful assistant. Answer the user's question using only "
            f"the provided context. If the context doesn't contain the answer, say so.\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            f"QUESTION: {query}\n\n"
            f"ANSWER:"
        )

    def query(self, query: str) -> RAGResponse:
        """Full RAG pipeline: retrieve → build prompt → generate → return."""
        context = self.retrieve(query)

        if self.llm_adapter:
            prompt = self._build_prompt(query, context)
            llm_response = self.llm_adapter.send(prompt)
            response_text = llm_response.content if llm_response.success else f"[LLM Error: {llm_response.error}]"
            model = self.llm_adapter.model
        else:
            # Simulation mode — return retrieved context as the "response"
            response_text = self._simulate_response(query, context)
            model = "simulation"

        # Detect if poisoned docs influenced the response
        was_poisoned = context.contains_poisoned
        poison_influence = None
        if was_poisoned:
            poisoned_docs = [d for d in context.documents if d.is_poisoned]
            poison_influence = f"{len(poisoned_docs)} poisoned doc(s) in context: " + \
                               ", ".join(d.poisoning_type or "unknown" for d in poisoned_docs)

        rag_response = RAGResponse(
            query=query,
            response=response_text,
            context=context,
            model=model,
            was_poisoned=was_poisoned,
            poison_influence=poison_influence
        )
        self.query_log.append(rag_response)
        return rag_response

    def _simulate_response(self, query: str, context: RetrievedContext) -> str:
        """Simulate LLM response for demo purposes (no API key needed)."""
        if not context.documents:
            return "No relevant documents found in knowledge base."

        if context.contains_poisoned:
            poisoned = [d for d in context.documents if d.is_poisoned]
            clean = [d for d in context.documents if not d.is_poisoned]
            sim = f"[SIMULATED RESPONSE — POISONED]\n"
            sim += f"Based on the retrieved context:\n\n"
            if clean:
                sim += f"Clean sources say: {clean[0].content[:150]}...\n\n"
            sim += f"⚠️ POISONED source ({poisoned[0].poisoning_type}) injected: {poisoned[0].content[:150]}..."
        else:
            sim = f"[SIMULATED RESPONSE — CLEAN]\n"
            sim += f"Based on retrieved context: {context.documents[0].content[:200]}..."
        return sim

    def stats(self) -> Dict:
        return {
            "total_documents": self.vector_store.document_count,
            "poisoned_documents": self.vector_store.poisoned_count,
            "total_queries": len(self.query_log),
            "poisoned_queries": sum(1 for r in self.query_log if r.was_poisoned),
        }
