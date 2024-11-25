from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

def build_vector_store(text: str):
    """
    Split text into chunks and create a FAISS vector store.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)

    return vector_store, chunks


def retrieve_relevant_chunks(vector_store, query: str, k=5):
    """
    Retrieve top-k relevant text chunks for the given query.
    """
    retriever = vector_store.as_retriever()
    relevant_chunks = retriever.get_relevant_documents(query)[:k]
    return " ".join([chunk.page_content for chunk in relevant_chunks])
