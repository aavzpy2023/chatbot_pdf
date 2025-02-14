import os

import PyPDF2
import streamlit as st
from transformers import pipeline

# Verificar la instalación de PyPDF2
st.write("PyPDF2 instalado correctamente:", PyPDF2.__version__)

# Cargar el modelo de lenguaje
model_name = "mistral:7b"  # Cambia esto al modelo que prefieras
qa_pipeline = pipeline("question-answering", model=model_name)


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text


def load_documents(folder_path):
    documents_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            documents_text += extract_text_from_pdf(pdf_path)
    return documents_text


def answer_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result["answer"]


# Cargar los documentos
documents_folder = "documents"
documents_text = load_documents(documents_folder)

# Interfaz de Streamlit
with st.sidebar:
    st.header("Proyecto")
    if st.button("Borrar"):
        st.rerun()
    st.header("Asistente virtual Versat.")
    user_input = st.chat_input("En que puedo ayudarte...")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            response = answer_question(user_input, documents_text)
            st.write(response)
