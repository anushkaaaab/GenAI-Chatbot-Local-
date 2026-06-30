import json
from typing import List
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)
chat_history = []

def partition_document(file_path: str):
    print("Loading...")
    elements = partition_pdf(
        filename= file_path,
        strategy= "hi_res",
        infer_table_structure= True,
        extract_image_block_types= ["Image"],
        extract_image_block_to_payload= True 
    )

    return elements

def create_chunks_by_title(elements):
    chunks = chunk_by_title(
        elements,
        max_characters= 3000,
        new_after_n_chars= 2400,
        combine_text_under_n_chars= 500
    )

    return chunks

def separate_content_types(chunk):
    content_data = {
        'text': chunk.text,
        'tables': [],
        'images': [],
        'types': ['text']
    }
    

    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__
            
            
            if element_type == 'Table':
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                content_data['tables'].append(table_html)
            
            
            elif element_type == 'Image':
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)
    
    content_data['types'] = list(set(content_data['types']))
    return content_data

def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    
    try:

        prompt_text = f"""
You are creating a searchable description for document retrieval.

TEXT
{text}

"""

        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}\n{table}\n\n"

        prompt_text += """
                YOUR TASK

                Generate a searchable summary that includes

                1. Key facts
                2. Main ideas
                3. Questions this chunk answers
                4. Description of tables/images
                5. Alternative search terms

                Only output the searchable description.
                """


        message_content=[
            {
                "type": "text",
                "text": prompt_text,
            },
            *[
                {
                   "type": "image",
                   "image": image_base64,
                }
                for image_base64 in images
            ]
    ]
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
                
        return response.content
        
    except Exception as e:
        #print(f"AI summary failed: {e}")
        
        summary = f"{text[:300]}..."
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary
    
def summarise_chunks(chunks):
    #print("Processing chunks with AI Summaries...")
    
    langchain_documents = []
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        current_chunk = i + 1
        #print(f"   Processing chunk {current_chunk}/{total_chunks}")
        

        content_data = separate_content_types(chunk)
        

        #print(f"     Types found: {content_data['types']}")
        #print(f"     Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}")
        

        if content_data['tables'] or content_data['images']:

            try:
                enhanced_content = create_ai_enhanced_summary(
                    content_data['text'],
                    content_data['tables'], 
                    content_data['images']
                )

            except Exception as e:
                enhanced_content = content_data['text']
        else:
            enhanced_content = content_data['text']
        

        doc = Document(
            page_content=enhanced_content,
            metadata={
                "original_content": json.dumps
                (
                    {
                        "raw_text": content_data['text'],
                        "tables_html": content_data['tables'],
                        "images_base64": content_data['images']
                    }
                )
            }
        )
        
        langchain_documents.append(doc)
    
    return langchain_documents


def export_chunks_to_json(chunks, filename="chunks_export.json"):
    export_data = []
    
    for i, doc in enumerate(chunks):
        chunk_data = {
            "chunk_id": i + 1,
            "enhanced_content": doc.page_content,
            "metadata": {
                "original_content": json.loads(doc.metadata.get("original_content", "{}"))
            }
        }
        export_data.append(chunk_data)
    

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    return export_data

def create_vector_store(documents, persist_directory="db/chroma_db"):
        
    embedding_model = embeddings

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )

    return vectorstore

def run_complete_ingestion_pipeline(pdf_path: str):
    
    elements = partition_document(pdf_path)
    chunks = create_chunks_by_title(elements)
    summarised_chunks = summarise_chunks(chunks)
    db = create_vector_store(summarised_chunks, persist_directory="db/chroma_db")
    
    return db

db = run_complete_ingestion_pipeline("./docs/attention-is-all-you-need.pdf")

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")
    
    if chat_history:
        messages = [
            SystemMessage(content= "Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        result = llm.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    
    else:
        search_question = user_question

    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)
    
    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        # Show first 2 lines of each document
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"  Doc {i}: {preview}...")

    combined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """

    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    result = llm.invoke(messages)
    answer = result.content

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    
    print(f"Answer: {answer}")
    return answer

def start_chat():
    print("Ask me questions! Type 'quit' to exit.")
    
    while True:
        question = input("\nYour question: ")
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
            
        ask_question(question)

if __name__ == "__main__":
    start_chat()
