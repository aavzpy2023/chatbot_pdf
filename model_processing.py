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
    "qwen2.5:1.5b": "Qwen2.5 1.5B (Detallado)",
    "qwen2.5:14b": "Qwen2.5 14B (Detallado)",
    "mistral:7b": "Mistral 7B (Rápido)",
}

ARCHIVO_CONTEXTO = "./documents/data_for_model.json"


# Prompt minimalista
PROMPT_TEMPLATE = """
Eres un asistente especializado en Versat Sarasola. Sigue estas instrucciones:

1. Si la pregunta es un saludo como "Hola", "Buenas", "Buenos días", etc., responde con un saludo amigable.
2. Si la pregunta es sobre Versat Sarasola, proporciona información basada en el contexto dado.
3. Si te preguntan sobre otros temas que no conoces, responde: "No tengo información".
4. Si te preguntan por los pasos de alguna accion, brinda absolutamente todos los pasos al usuario. Prohibido omitir ningun paso.
5. Si la respuesta es corta, responde con una frase corta.
6. Si la respuesta no puede ser proporcionada basada en el contexto, responde: "No tengo información".
7. Si la pregunta menciona "usuario regular", "usuario normal" o "empleado", **ignora la sección ADMINISTRADOR**.
8. Si la pregunta menciona "administrador", "configuración inicial" o "usuario 'sa'", **ignora la sección USUARIO_REGULAR**.

Contexto:
{context}

Pregunta:
{question}

Respuesta:
"""


# Clase personalizada para embeddings usando Ollama
class OllamaEmbeddings(Embeddings):
    def __init__(
        self,
        model: str = "nomic-embed-text:latest",
        embedding_dim: int = 768,
        max_retries: int = 3,
        fallback_type: str = "zero",
    ):
        self.model = model
        self.embedding_dim = embedding_dim
        self.max_retries = max_retries
        self.fallback_type = fallback_type  # Opciones: "zero", "random", "ones"

    def embed_documents(self, texts):
        # Return a list of flattened embedding vectors
        # print("Embedding documents:", texts)
        if not texts:
            return []

        embeddings = []

        for text in texts:
            result = ollama.embed(self.model, text)
            # Safely access the first embedding, if available
            if result["embeddings"] and len(result["embeddings"]) > 0:
                embeddings.append(result["embeddings"][0])
            else:
                # Log or handle the case when no embeddings are returned
                print(f"Warning: No embeddings returned for text: {text[:30]}...")
                # Add a zero vector as placeholder for failed embeddings
                # Ensure same dimensions as other embeddings
                embeddings.append([0.0] * self.embedding_dim)  # Dimensión configurable

        return embeddings

    def embed_query(self, text):
        # Return a single flattened embedding vector
        import random
        import logging
        import time

        # Intentar generar embedding con reintentos
        for attempt in range(self.max_retries):
            try:
                result = ollama.embed(self.model, text)
                if result.get("embeddings") and len(result["embeddings"]) > 0:
                    return result["embeddings"][0]

                # Si no hay embeddings pero la llamada no falló, intentamos de nuevo
                logging.warning(
                    f"Intento {attempt+1}/{self.max_retries}: No se obtuvieron embeddings para: '{text[:50]}...'"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(0.5)  # Pequeña pausa antes de reintentar
                    continue
            except Exception as e:
                logging.error(
                    f"Error al generar embedding (intento {attempt+1}/{self.max_retries}): {str(e)}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(1)  # Pausa más larga si hay una excepción
                    continue

        # Si llegamos aquí, todos los intentos fallaron
        logging.warning(
            f"Todos los intentos de embedding fallaron para: '{text[:50]}...' - Usando vector de respaldo tipo '{self.fallback_type}'"
        )

        # Generar vector de respaldo según el tipo configurado
        if self.fallback_type == "random":
            return [random.uniform(-1.0, 1.0) for _ in range(self.embedding_dim)]
        elif self.fallback_type == "ones":
            return [1.0] * self.embedding_dim
        else:  # "zero" es el valor predeterminado
            return [0.0] * self.embedding_dim


@st.cache_resource
def crear_vector_store(chat_history: str = ""):
    """
    Create a vector store using chat history and embeddings.

    Args:
        chat_history (str): Historial de chat.

    Returns:
        FAISS: Vector store.
    """
    if not os.path.exists(ARCHIVO_CONTEXTO):
        st.error(f"Archivo {ARCHIVO_CONTEXTO} no encontrado")
        return None

    with open(ARCHIVO_CONTEXTO, "r", encoding="utf-8") as f:
        texto_completo = f.read()

    # Chunking optimizado
    overlap = 500
    chunks = []
    if chat_history:
        chunks = [
            texto_completo[i : i + overlap]
            for i in range(0, len("\n".join([chat_history, texto_completo])), overlap)
        ]

    # Split by specific sections
        chunks = []
        for seccion in ["ADMINISTRADOR", "USUARIO_REGULAR"]:
            if seccion in texto_completo:
                contenido = texto_completo.split(seccion)[1].split("}")[0]
                chunks.append(f"**{seccion}**{contenido}")


    current_chunk = ""
    for linea in texto_completo.split("\n"):
        if linea.strip().startswith("1. ") or linea.strip().startswith("2. "):
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
        current_chunk += linea + "\n"
    if current_chunk:
        chunks.append(current_chunk)
    # Usa el modelo de embeddings local
    embeddings_model = OllamaEmbeddings(model="nomic-embed-text:latest")
    print("embeddings:", embeddings_model)

    # Get embeddings
    embeddings = embeddings_model.embed_documents(chunks)

    if not embeddings:
        st.error("No se pudieron generar embeddings válidos para ningún texto")
        return None

    # Create FAISS using from_texts to properly handle the texts and embeddings
    vector_store = FAISS.from_texts(chunks, embeddings_model)
    print("FAISS: ", vector_store)
    return vector_store


# def create_model(
#     selected_model: str,
#     temperature=0,
#     base_url="http://localhost:11434",
#     timeout=120,
#     num_ctx=4096,
#     num_gpu=1,
# ):
#     """
#     Configure the model

#     arg selected_model (str): name of ollama model selected
#     """
#     llm = OllamaLLM(
#         model=selected_model,
#         temperature=temperature,
#         base_url=base_url,
#         # timeout=timeout,
#         num_ctx=num_ctx,
#         num_gpu=num_gpu,
#     )
#     return llm



def create_model(selected_model: str):
    llm = OllamaLLM(
        model=selected_model,
        temperature=float(os.getenv("OLLAMA_TEMPERATURE", 0.0)),  # Default: 0.0
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        # timeout=int(os.getenv("OLLAMA_TIMEOUT", 120)),           # Default: 120
        num_ctx=int(os.getenv("OLLAMA_NUM_CTX", 4096)),          # Default: 4096
        num_gpu=int(os.getenv("OLLAMA_NUM_GPU", 1))              # Default: 1
    )
    return llm
