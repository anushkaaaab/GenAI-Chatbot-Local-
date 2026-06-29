from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

persistent_directory = "db/chroma_db"

embedding_model = OllamaEmbeddings(model = "mxbai-embed-large")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

print("Enter query: ")
query = input()

retriever = db.as_retriever(search_kwargs={"k": 1})

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 1,
        "score_threshold": 0.75  # Only return chunks with cosine similarity ≥ 0.75
    }
)

relevant_docs = retriever.invoke(query)

#print("--- Context ---")
#for i, doc in enumerate(relevant_docs, 1):
#    print(f"Document {i}:\n{doc.page_content}\n")

combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

model = ChatOllama(model = "llama3.2")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

result = model.invoke(messages)

print("\n--- Generated Response ---")
print(result.content)