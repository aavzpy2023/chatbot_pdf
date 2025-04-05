import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import ollama
from langchain_ollama import OllamaLLM

@st.cache_data
def get_downloaded_models():
    """
    Get the downloaded models with ollama.

    This function will return a list of models available locally.

    Returns:
        list: A list of model names or an empty list if no models are available.
    """
    try:
        # Get model's list from ollama
        response = ollama.list()
        models_list = response.get("models", [])

        if not models_list:
            print("No models available.")
            return []

        # get name of the models
        models_name = [model['model'] for model in models_list]

    except Exception as e:
        print(f"Error retrieving models: {e}")
        models_name = []  # Return an empty list in case of error

    return models_name


ARCHIVO_CONTEXTO = "./documents/data_for_model.json"


# Minimalist prompt
PROMPT_TEMPLATE = """
Eres un asistente especializado en Versat Sarasola. Sigue estas instrucciones:

1. Si la pregunta es un saludo como "Hola", "Buenas", "Buenos días", etc., responde con un saludo amigable.
2. Si la pregunta es sobre Versat Sarasola, proporciona información basada en el contexto dado.
3. Si te preguntan sobre otros temas que no conoces, responde: "No tengo información".
4. Si te preguntan por los pasos de alguna acción, brinda absolutamente todos los pasos al usuario. Prohibido omitir ningún paso.
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

# Custom class for embeddings using Ollama
class OllamaEmbeddings(Embeddings):
    def __init__(
        self,
        model: str = "nomic-embed-text:latest",
        embedding_dim: int = 768,
        max_retries: int = 3,
        fallback_type: str = "zero",
    ):
        """
        Initialize the OllamaEmbeddings instance.

        Args:
            model (str): Name of the Ollama model to use for embeddings.
            embedding_dim (int): Dimension of the embedding vectors.
            max_retries (int): Maximum number of retries on failed embedding attempts.
            fallback_type (str): Type of fallback vector to generate if an error occurs. Options: "zero", "random", "ones".
        """
        self.model = model
        self.embedding_dim = embedding_dim
        self.max_retries = max_retries
        self.fallback_type = fallback_type  # Options: "zero", "random", "ones"

    def embed_documents(self, texts):
        """
        Embed a list of documents using Ollama.

        Args:
            texts (list): List of text strings to be embedded.

        Returns:
            list: A list of embedding vectors.
        """
        if not texts:
            return []
        embeddings = []
        for text in texts:
            result = ollama.embed(self.model, text)
            # Safely access the first embedding, if available
            if result["embeddings"] and len(result["embeddings"]) > 0:
                embeddings.append(result["embeddings"][0])
            else:
                print(f"Warning: No embeddings returned for text: {text[:30]}...")
                # Add a zero vector as placeholder for failed embeddings
                embeddings.append([0.0] * self.embedding_dim)  # Dimension configurable
        return embeddings

    def embed_query(self, text):
        """
        Embed a single query using Ollama.

        Args:
            text (str): Query string to be embedded.

        Returns:
            list: A single embedding vector.
        """
        import random
        import logging
        import time

        # Attempt to generate embedding with retries
        for attempt in range(self.max_retries):
            try:
                result = ollama.embed(self.model, text)
                if result.get("embeddings") and len(result["embeddings"]) > 0:
                    return result["embeddings"][0]

                logging.warning(
                    f"Intento {attempt+1}/{self.max_retries}: No se obtuvieron embeddings para: '{text[:50]}...'"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(0.5)  # Small pause before retry
                    continue
            except Exception as e:
                logging.error(
                    f"Error al generar embedding (intento {attempt+1}/{self.max_retries}): {str(e)}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(1)  # Longer pause if an exception occurs
                    continue

        logging.warning(
            f"Todos los intentos de embedding fallaron para: '{text[:50]}...' - Usando vector de respaldo tipo '{self.fallback_type}'"
        )

        # Generate a fallback vector based on the configured type
        if self.fallback_type == "random":
            return [random.uniform(-1.0, 1.0) for _ in range(self.embedding_dim)]
        elif self.fallback_type == "ones":
            return [1.0] * self.embedding_dim
        else:
            return [0.0] * self.embedding_dim


@st.cache_resource
def crear_vector_store(chat_history: str = ""):
    """
    Create a vector store using chat history and embeddings.

    Args:
        chat_history (str): Historial de chat.

    Returns:
        FAISS: Vector store or None if the archivo_conteo does not exist.
    """
    if not os.path.exists(ARCHIVO_CONTEXTO):
        st.error(f"Archivo {ARCHIVO_CONTEXTO} no encontrado")
        return None

    with open(ARCHIVO_CONTEXTO, "r", encoding="utf-8") as f:
        texto_completo = f.read()

    # Optimized Chunking
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
    # print("embeddings:", embeddings_model)

    # Get embeddings
    embeddings = embeddings_model.embed_documents(chunks)

    if not embeddings:
        st.error("No se pudieron generar embeddings válidos para ningún texto")
        return None

    # Create FAISS using from_texts to properly handle the texts and embeddings
    vector_store = FAISS.from_texts(chunks, embeddings_model)
    # print("FAISS: ", vector_store)
    return vector_store


def create_model(selected_model: str):
    """
    Configure and create an OllamaLLM instance with default or environment-configured settings.

    Args:
        selected_model (str): Name of the ollama model to use.

    Returns:
        OllamaLLM: An initialized OllamaLLM instance.
    """
    llm = OllamaLLM(
        model=selected_model,
        temperature=float(os.getenv("OLLAMA_TEMPERATURE", 0.0)),  # Default: 0.0
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        num_ctx=int(os.getenv("OLLAMA_NUM_CTX", 4096)),          # Default: 4096
        num_gpu=int(os.getenv("OLLAMA_NUM_GPU", 1))              # Default: 1
    )
    return llm
