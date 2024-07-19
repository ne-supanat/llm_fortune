from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_db():
    db = Chroma(
        embedding_function=OllamaEmbeddings(
            model="mxbai-embed-large", show_progress=True
        ),
        collection_name="tarot_guide",
        persist_directory="./chroma_db",
    )

    if db._collection.count() > 0:
        return db
    else:
        convert_pdf_to_text_langchain()
        return get_db()


def convert_pdf_to_text_langchain():
    ## Document load
    loader = PyPDFLoader("The Ultimate Guide to Tarot - A Beginner.pdf")
    data = loader.load()

    ## Text split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data[:10])

    ## Embedding & Vector store
    Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="mxbai-embed-large", show_progress=True),
        collection_name="tarot_guide",
        persist_directory="./chroma_db",
    )


def delete_collection():
    Chroma(
        collection_name="tarot_guide",
        persist_directory="./chroma_db",
    ).delete_collection()


if __name__ == "__main__":
    delete_collection()
    convert_pdf_to_text_langchain()
