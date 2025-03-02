import os
import streamlit as st
# from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import ollama
from langchain_ollama import OllamaLLM

# Configuración
MODELOS = {
    "qwen2.5-coder:7b": "Qwen2.5 Coder 7B (Detallado)",
    "mistral:7b": "Mistral 7B (Rápido)",
}

ARCHIVO_CONTEXTO = "context2.txt"


# Prompt minimalista
PROMPT_TEMPLATE = """
Eres un asistente especializado en Versat Sarasola. Sigue estas instrucciones:

1. Si la pregunta es un saludo como "Hola", "Buenas", "Buenos días", etc., responde con un saludo amigable.
2. Si la pregunta es sobre Versat Sarasola, proporciona información basada en el contexto dado.
3. Si te preguntan sobre otros temas que no conoces, responde: "No tengo información".
4. Si la respuesta es corta, responde con una frase corta.
5. Si la respuesta no puede ser proporcionada basada en el contexto, responde: "No tengo información".

Contexto:
{context}

Pregunta:
{question}

Respuesta:
"""


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
    chunks = [texto_completo[i:i+300]
              for i in range(0, len(texto_completo), 300)]

    # Usa el modelo de embeddings local
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    return FAISS.from_texts(chunks, embeddings)


def create_model(selected_model: str, temperature=0, base_url="http://localhost:11434",
                 timeout=120, num_ctx=2048, num_gpu=1):
    """
    Configure the model

    arg selected_model (str): name of ollama model selected
    """
    llm = OllamaLLM(
        model=selected_model,
        temperature=temperature,
        base_url=base_url,
        # timeout=timeout,
        num_ctx=num_ctx,
        num_gpu=num_gpu
    )
    return llm
