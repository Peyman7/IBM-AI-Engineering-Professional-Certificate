from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from llm_utils import get_llm, get_embedding_model
from document_utils import load_document, split_document

def get_retriever(file_path):
    docs = load_document(file_path)
    chunks = split_document(docs)
    embeddings = get_embedding_model()
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="doc_collection"
    )
    return vectordb.as_retriever()

def run_qa(file_path, query):
    retriever = get_retriever(file_path)
    llm = get_llm()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )
    result = qa.invoke(query)
    return result['result']
