import os
import streamlit as st
import json
import base64
from ingestion_pipeline import run_complete_ingestion_pipeline
from multimodal_chat import load_vectorstore, ask_question

st.set_page_config(
    page_title= "Chatbot",
    layout= "wide"
)

st.title("Chatbot")

DB_PATH = "db/chroma_db"

if not os.path.exists(DB_PATH):
    with st.spinner("Preparing the knowledge base..."):
        run_complete_ingestion_pipeline(
            "./docs/attention-is-all-you-need.pdf"
        )

if "db" not in st.session_state:
    st.session_state.db = load_vectorstore()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:

    role = "user"

    if message.__class__.__name__ == "AIMessage":
        role = "assistant"

    with st.chat_message(role):
        st.write(message.content)

question = st.chat_input(
    "Ask me a question!"
)

if question:

    with st.chat_message("user"):
        st.write(question)

    with st.spinner("Thinking..."):
        answer, docs = ask_question(
        question,
        st.session_state.db,
        st.session_state.chat_history,
    )

    with st.chat_message("assistant"):
        st.write(answer)
    
    with st.sidebar:

        st.header("Retrieved Chunks")

        for i, doc in enumerate(docs, 1):

            with st.expander(f"Chunk {i}"):
                st.write(doc.page_content)

            metadata = json.loads(doc.metadata["original_content"])

            for image_base64 in metadata["images_base64"]:
                image_bytes = base64.b64decode(image_base64)
                st.image(image_bytes)
                
st.sidebar.success("Knowledge base ready!")



