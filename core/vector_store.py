import os 
from langchain_chroma import Chroma 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import shutil

CHROMA_DIR = "vector_db"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL,
        model_kwargs = {"device" : 'cpu'}
    )

def build_vector_store(transcript : str, meeting_id: str = "default_meeting")->Chroma:

    print("Building vector Store")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata = {'chunk_index' : i, "meeting_id": meeting_id})
        for i,chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents= docs,
        embedding=embeddings,
        collection_name=meeting_id,
        persist_directory=CHROMA_DIR
    )

    return vector_store



def load_vector_store(meeting_id: str = "default_meeting") ->Chroma:
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=meeting_id,
        embedding_function= embeddings,
        persist_directory=CHROMA_DIR
    )

    return vector_store

def delete_vector_store_collection(meeting_id: str = "default_meeting",) -> bool:
    """Permanently removes an isolated meeting collection from the Chroma vector database."""
    try:
        vs = load_vector_store(meeting_id)
        vs._client.delete_collection(name=meeting_id)
        return True
    except Exception as e:
        print(f"Error deleting collection {meeting_id}: {e}")
        return False


def get_retriever(vector_store : Chroma, k :int = 4):
    return vector_store.as_retriever(
        search_type = 'similarity',
        search_kwargs = {"k":k}
    )


