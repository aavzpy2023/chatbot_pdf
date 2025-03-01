import os
import streamlit as st
# import numpy as np
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import ollama

# Configuración
MODELOS = {
    "qwen2.5-coder:7b": "Qwen2.5 Coder 7B (Detallado)",
    "mistral:7b": "Mistral 7B (Rápido)",
}

ARCHIVO_CONTEXTO = "context.txt"

# Clase personalizada para embeddings usando Ollama
class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str = "nomic-embed-text:latest"):
        self.model = model

    def embed_documents(self, texts):
        # Return a list of flattened embedding vectors
        return [ollama.embed(self.model, text)["embeddings"][0] for text in texts]

    def embed_query(self, text):
        # Return a single flattened embedding vector
        return ollama.embed(self.model, text)["embeddings"][0]

@st.cache_resource
def crear_vector_store():
    if not os.path.exists(ARCHIVO_CONTEXTO):
        st.error(f"Archivo {ARCHIVO_CONTEXTO} no encontrado")
        return None

    with open(ARCHIVO_CONTEXTO, "r", encoding="utf-8") as f:
        texto_completo = f.read()

    # Chunking optimizado
    chunks = [texto_completo[i:i+300] for i in range(0, len(texto_completo), 300)]

    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")  # Usa el modelo de embeddings local
    return FAISS.from_texts(chunks, embeddings)

# Prompt minimalista
PROMPT_TEMPLATE = """
Eres un asistente especializado en Versat Sarasola. Sigue estas instrucciones:

1. Si la pregunta es un saludo como "Hola", "Buenas", "Buenos días", etc., responde con un saludo amigable.
2. Si la pregunta es sobre Versat Sarasola, proporciona información basada en el contexto dado.
3. Si te preguntan sobre otros temas que no conoces, responde: "No tengo información".

Contexto:
{context}

Pregunta:
{question}

Respuesta:
"""

def main():
    st.set_page_config(page_title="Asistente Versat Sarasola", layout="wide")
    st.title("🤖 Asistente Versat Sarasola")

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    # Sidebar con modelos disponibles
    with st.sidebar:
        modelo_seleccionado = st.selectbox(
            "Modelo:",
            options=list(MODELOS.keys()),
            format_func=lambda x: MODELOS[x],
            index=0  # Modelo por defecto: Qwen2.5 Coder 7B
        )

    # Configuración del LLM con Ollama
    llm = Ollama(
        model=modelo_seleccionado,
        temperature=0.1,
        base_url="http://localhost:11434",
        timeout=60,
        num_ctx=4096,
        num_gpu=1  # Si tienes GPU
    )

    # Crear QA chain
    vector_store = crear_vector_store()
    if vector_store:
        PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
        )

    # Interfaz de chat
    if "historial" not in st.session_state:
        st.session_state.historial = []

    for mensaje in st.session_state.historial:
        with st.chat_message(mensaje["role"]):
            st.write(mensaje["content"])

    pregunta = st.chat_input("Escribe tu pregunta sobre Versat Sarasola...")

    if pregunta and st.session_state.qa_chain:
        with st.chat_message("user"):
            st.write(pregunta)

        try:
            with st.spinner("Analizando..."):
                respuesta = st.session_state.qa_chain.invoke(pregunta)
                respuesta_limpia = respuesta.get("result", "No disponible")

            with st.chat_message("assistant"):
                st.write(respuesta_limpia)

            st.session_state.historial.extend([
                {"role": "user", "content": pregunta},
                {"role": "assistant", "content": respuesta_limpia},
            ])

        except Exception as e:
            st.error("Error: Reinicia el servidor de Ollama" + str(e))
            st.session_state.historial.append({
                "role": "assistant",
                "content": "Ocurrió un error. Verifica que Ollama esté activo."
            })

if __name__ == "__main__":
    main()
