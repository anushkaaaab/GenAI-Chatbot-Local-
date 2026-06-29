import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist.")
    
    loader = DirectoryLoader (
        path = docs_path,
        glob = "**/*.txt",
        loader_cls = TextLoader,
        loader_kwargs={
        "encoding": "utf-8",
        "autodetect_encoding": True
    }
    )
    
    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}.")
    

    for i, doc in enumerate(documents): 
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size = 800, chunk_overlap = 0):

    textSplitter = CharacterTextSplitter (
        chunk_overlap = chunk_overlap,
        chunk_size = chunk_size
    )

    chunks = textSplitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks

def create_vector_store(chunks, persist_directory = "db/chroma_db"):

    print("Creating embeddings and storing in ChromaDB...")
    embedding_model = OllamaEmbeddings(model = "mxbai-embed-large")

    print("--- Creating vector store ---")
    vectorStore = Chroma(
        embedding_function = embedding_model,
        persist_directory = persist_directory,
        collection_metadata= {"hnsw:space":"cosine"}
    )

    batch_size = 100

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        print(f"Adding batch {i//batch_size + 1}")
        vectorStore.add_documents(batch)


    print("--- Finished creating vector store ---")
    print(f"Vector store created and saved to {persist_directory}")

    return vectorStore

def main():
    print("Main Function")
    docs_path = "docs" 
    persistent_directory = "db/chroma_db"
    
    documents = load_documents(docs_path)
    chunks = split_documents(documents)
    vectorStore = create_vector_store(chunks, persistent_directory)
    
    return vectorStore

if __name__ == "__main__":
    main()


