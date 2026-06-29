import json
from typing import List
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

def partition_document(file_path: str):
    print("Loading...")
    #print(f"Partioning document: {file_path}...")
    elements = partition_pdf(
        filename= file_path,
        strategy= "hi_res",
        infer_table_structure= True,
        extract_image_block_types= ["Image"],
        extract_image_block_to_payload= True 
    )

    #print(f"---Extracted {len(elements)} elements---")
    return elements
file_path = "./docs/attention-is-all-you-need.pdf"
elements = partition_document(file_path)
elements

set([str(type(el)) for el in elements])
elements[36].to_dict()

images = [element for element in elements if element.category == 'Image']
#print(f"---Found {len(images)} images---")

images[0].to_dict()

tables = [element for element in elements if element.category == 'Table']
#print(f"---Found {len(tables)} tables---")

tables[0].to_dict()

def create_chunks_by_title(elements):
    #print("Creating chunks...")
    chunks = chunk_by_title(
        elements,
        max_characters= 3000,
        new_after_n_chars= 2400,
        combine_text_under_n_chars= 500
    )

    #print(f"---Created {len(chunks)} chunks---")
    return chunks

chunks = create_chunks_by_title(elements)

set([str(type(chunk)) for chunk in chunks])
chunks[11].metadata.orig_elements[-1].to_dict()

def separate_content_types(chunk):
    """Analyze what types of content are in a chunk"""
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

        llm = ChatOllama(model="llama3.2", temperature=0)
        

        prompt_text = f"""You are creating a searchable description for document content retrieval.

        CONTENT TO ANALYZE:
        TEXT CONTENT:
        {text}

        """
        

        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n\n"
        
                prompt_text += """
                YOUR TASK:
                Generate a comprehensive, searchable description that covers:

                1. Key facts, numbers, and data points from text and tables
                2. Main topics and concepts discussed  
                3. Questions this content could answer
                4. Visual content analysis (charts, diagrams, patterns in images)
                5. Alternative search terms users might use

                Make it detailed and searchable - prioritize findability over brevity.

                SEARCHABLE DESCRIPTION:"""


        message_content = [{"type": "text", "text": prompt_text}]
        

        for image_base64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        
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
            #print(f"     → Creating AI summary for mixed content...")
            try:
                enhanced_content = create_ai_enhanced_summary(
                    content_data['text'],
                    content_data['tables'], 
                    content_data['images']
                )
                #print(f"     → AI summary created successfully")
                #print(f"     → Enhanced content preview: {enhanced_content[:200]}...")
            except Exception as e:
                #print(f"AI summary failed: {e}")
                enhanced_content = content_data['text']
        else:
            #print(f"     → Using raw text (no tables/images)")
            enhanced_content = content_data['text']
        

        doc = Document(
            page_content=enhanced_content,
            metadata={
                "original_content": json.dumps({
                    "raw_text": content_data['text'],
                    "tables_html": content_data['tables'],
                    "images_base64": content_data['images']
                })
            }
        )
        
        langchain_documents.append(doc)
    
    #print(f"Processed {len(langchain_documents)} chunks")
    return langchain_documents


processed_chunks = summarise_chunks(chunks)
processed_chunks

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
    
    #print(f"Exported {len(export_data)} chunks to {filename}")
    return export_data


json_data = export_chunks_to_json(processed_chunks)

def create_vector_store(documents, persist_directory="dbv1/chroma_db"):
    #print("Creating embeddings and storing in ChromaDB...")
        
    embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
    

    #print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    #print("--- Finished creating vector store ---")
    
    #print(f"Vector store created and saved to {persist_directory}")
    return vectorstore


db = create_vector_store(processed_chunks)

print("Enter query:")
query = input()
retriever = db.as_retriever(search_kwargs={"k": 3})
chunks = retriever.invoke(query)
export_chunks_to_json(chunks, "rag_results.json")

def run_complete_ingestion_pipeline(pdf_path: str):
    """Run the complete RAG ingestion pipeline"""
    #print("Starting RAG Ingestion Pipeline")
    #print("=" * 50)
    
    elements = partition_document(pdf_path)
    chunks = create_chunks_by_title(elements)
    summarised_chunks = summarise_chunks(chunks)
    db = create_vector_store(summarised_chunks, persist_directory="dbv2/chroma_db")
    
    #print("Pipeline completed successfully!")
    return db

db = run_complete_ingestion_pipeline("./docs/attention-is-all-you-need.pdf")

def generate_final_answer(chunks, query):
    try:

        llm = ChatOllama(model="llama3.2", temperature=0)
        
        prompt_text = f"""Based on the following documents, please answer this question: {query}

CONTENT TO ANALYZE:
"""
        
        for i, chunk in enumerate(chunks):
            prompt_text += f"--- Document {i+1} ---\n"
            
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                
                
                raw_text = original_data.get("raw_text", "")
                if raw_text:
                    prompt_text += f"TEXT:\n{raw_text}\n\n"
                
                
                tables_html = original_data.get("tables_html", [])
                if tables_html:
                    prompt_text += "TABLES:\n"
                    for j, table in enumerate(tables_html):
                        prompt_text += f"Table {j+1}:\n{table}\n\n"
            
            prompt_text += "\n"
        
        prompt_text += """
Please provide a clear, comprehensive answer using the text, tables, and images above. If the documents don't contain sufficient information to answer the question, say "I don't have enough information to answer that question based on the provided documents."

ANSWER:"""

        message_content = [{"type": "text", "text": prompt_text}]
        
        for chunk in chunks:
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                images_base64 = original_data.get("images_base64", [])
                
                for image_base64 in images_base64:
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    })
        

        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        
        return response.content
        
    except Exception as e:
        #print(f"Answer generation failed: {e}")
        return "Sorry, I encountered an error while generating the answer."

final_answer = generate_final_answer(chunks, query)
print(final_answer)