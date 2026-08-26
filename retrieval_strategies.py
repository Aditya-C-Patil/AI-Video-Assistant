import os
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"}
    )


# 1. Simple Fixed-Size Chunking + Dense Search
def build_simple_retriever(transcript: str, meeting_id: str = "simple_eval"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(transcript)
    docs = [
        Document(page_content=c, metadata={"meeting_id": meeting_id})
        for c in chunks
    ]

    vs = Chroma.from_documents(
        docs, embedding=get_embeddings(), collection_name=f"{meeting_id}_simple"
    )
    return vs.as_retriever(search_kwargs={"k": 3})


# 2. Semantic Chunking
def build_semantic_retriever(
    transcript: str, meeting_id: str = "semantic_eval"
):
    embeddings = get_embeddings()
    semantic_splitter = SemanticChunker(
        embeddings, breakpoint_threshold_type="percentile"
    )
    docs = semantic_splitter.create_documents([transcript])
    for d in docs:
        d.metadata["meeting_id"] = meeting_id

    vs = Chroma.from_documents(
        docs, embedding=embeddings, collection_name=f"{meeting_id}_semantic"
    )
    return vs.as_retriever(search_kwargs={"k": 3})


# 3. Hybrid Search (Native BM25 + Dense RRF Fusion)
class NativeHybridRetriever(BaseRetriever):
    docs: List[Document]
    chroma_retriever: BaseRetriever
    k: int = 3

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # 1. Dense Search
        dense_docs = self.chroma_retriever.invoke(query)

        # 2. BM25 Search
        tokenized_corpus = [doc.page_content.lower().split() for doc in self.docs]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(query.lower().split())
        top_bm25_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[: self.k]
        bm25_docs = [self.docs[i] for i in top_bm25_indices]

        # 3. Reciprocal Rank Fusion (RRF)
        doc_scores = {}
        for rank, doc in enumerate(dense_docs):
            doc_scores[doc.page_content] = doc_scores.get(
                doc.page_content, 0
            ) + 1.0 / (60 + rank + 1)
        for rank, doc in enumerate(bm25_docs):
            doc_scores[doc.page_content] = doc_scores.get(
                doc.page_content, 0
            ) + 1.0 / (60 + rank + 1)

        sorted_docs = sorted(
            doc_scores.items(), key=lambda x: x[1], reverse=True
        )
        final_texts = [text for text, _ in sorted_docs[: self.k]]

        return [Document(page_content=t) for t in final_texts]


def build_hybrid_retriever(transcript: str, meeting_id: str = "hybrid_eval"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(transcript)
    docs = [
        Document(page_content=c, metadata={"meeting_id": meeting_id})
        for c in chunks
    ]

    vs = Chroma.from_documents(
        docs, embedding=get_embeddings(), collection_name=f"{meeting_id}_hybrid"
    )
    dense_retriever = vs.as_retriever(search_kwargs={"k": 3})

    return NativeHybridRetriever(
        docs=docs, chroma_retriever=dense_retriever, k=3
    )


# 4. Reranking (Native FlashRank Cross-Encoder)
class NativeRerankRetriever(BaseRetriever):
    chroma_retriever: BaseRetriever
    k: int = 3

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        base_docs = self.chroma_retriever.invoke(query)
        try:
            from flashrank import Ranker, RerankRequest

            ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp")
            passages = [
                {"id": idx, "text": d.page_content}
                for idx, d in enumerate(base_docs)
            ]
            rerank_request = RerankRequest(query=query, passages=passages)
            results = ranker.rerank(rerank_request)

            top_results = results[: self.k]
            return [Document(page_content=r["text"]) for r in top_results]
        except Exception:
            return base_docs[: self.k]


def build_reranked_retriever(
    transcript: str, meeting_id: str = "rerank_eval"
):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(transcript)
    docs = [
        Document(page_content=c, metadata={"meeting_id": meeting_id})
        for c in chunks
    ]

    vs = Chroma.from_documents(
        docs, embedding=get_embeddings(), collection_name=f"{meeting_id}_rerank"
    )
    dense_retriever = vs.as_retriever(search_kwargs={"k": 8})

    return NativeRerankRetriever(chroma_retriever=dense_retriever, k=3)
