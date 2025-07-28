from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from config import CHUNK_SIZE, CHUNK_OVERLAP

def load_document(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

def split_document(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_documents(documents)
