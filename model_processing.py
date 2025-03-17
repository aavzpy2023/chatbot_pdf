from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pickle
import re
from dotenv import load_dotenv
import os
import datetime
import streamlit as st


load_dotenv()

# Plantilla de prompt personalizada
PROMPT_TEMPLATE = """
Eres un asistente especializado en Versat Sarasola. Sigue estas instrucciones:

1. Si la pregunta es un saludo como "Hola", "Buenas", "Buenos días", etc., responde con un saludo amigable.
2. Si la pregunta es sobre Versat Sarasola, proporciona información basada solo en el contexto dado.
3. Si te preguntan sobre otros temas que no conoces, responde: "No tengo información".
4. Si te preguntan por los pasos de alguna acción, brinda absolutamente todos los pasos al usuario. Prohibido omitir ningún paso.
5. Si la respuesta es corta, responde con una frase corta.
6. Si la respuesta no puede ser proporcionada basada en el contexto, responde: "No tengo información".
7. Si la pregunta menciona "usuario regular", "usuario normal" o "empleado", **ignora la sección ADMINISTRADOR**.
8. Si la pregunta menciona "administrador", "configuración inicial" o "usuario 'sa'", **ignora la sección USUARIO_REGULAR**.
9. Brinda toda la informacion posible al usuario de acuerdo con los datos brindados para cada una de las preguntas.
10. A la hora de responder al usuario No considere las secciones de las preguntas que no contenian informacion relevante
11. Si no se tiene informacion de algun aspecto de la pregunta no mencionarlo al usuario. Esta prohibido
12. Si no tienes informacion suficiente sobre un tema, evita mencionar al usuario que no hay mas informacion en los documentos.
13. Cuando respondas, hazlo a partir del contenido y si no hay suficiente informacion solo responde y no menciones que no hay mas informacion.


Contexto:
{context}

Pregunta:
{question}

"""

# =============================================================================
# Función para leer el documento
# =============================================================================
def load_document(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print_with_date(f"❌ Error: El archivo '{file_path}' no se encontró.")
        return None
    except Exception as e:
        print_with_date(f"❌ Error al leer el archivo: {e}")
        return None

# =============================================================================
# Función para extraer detalles de un bloque de texto
# =============================================================================
def extract_details(content: str):
    """
    Extract information from a content and structure it
    """
    data = {}
    for section in content.split("Sct."):
        if ":" in section:
            splits = section.split(":")
            data[splits[0].lower()] = splits[1]
        else:
            data["id"] = section
    return data


# =============================================================================
# Función para dividir el documento en chunks basados en preguntas
# =============================================================================
def parse_document(text):
    """
    Parse documents of data
    """
    chunks = []
    try:
        # Dividir el texto en bloques basados en "ID:"
        preguntas = re.split('ID: ', text, flags=re.IGNORECASE)

        print_with_date(f"Un total de {len(preguntas) - 1} preguntas fueron detectadas.")

        for pregunta in preguntas:

            if ".CU" in pregunta:
                id_titulo = pregunta.strip()  # "ID: ..."
                contenido = pregunta.strip()  # Contenido asociado

                # Extraer detalles del contenido
                detalles = extract_details(id_titulo + "\n" + contenido)

                chunk = ''
                for k, v in detalles.items():
                    chunk += ":".join([k, v]) + "\n"



                chunks.append(chunk)

    except Exception as e:
        print_with_date(f"❌ Error procesando el documento: {e}")

    return chunks



def create_model(selected_model: str):
    """
    Configure and create an OllamaLLM instance with default or environment-configured settings.

    Args:
        selected_model (str): Name of the ollama model to use.

    Returns:
        OllamaLLM: An initialized OllamaLLM instance.
    """
    temperature = float(os.getenv("OLLAMA_TEMPERATURE", 0.0))
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    num_ctx=int(os.getenv("OLLAMA_NUM_CTX", 4096))
    num_gpu=int(os.getenv("OLLAMA_NUM_GPU", 1))



    llm = OllamaLLM(
        model=selected_model,
        temperature=temperature,
        base_url=base_url,
        num_ctx=num_ctx,
        num_gpu=num_gpu,
    )
    return llm


def print_with_date(message: str):
    print(datetime.datetime.now(), "-->", message)


@st.cache_data
def load_and_process_document(file_path):
    """
    Loads and processes the document into chunks for further use.

    Args:
        file_path (str): Path to the document file.

    Returns:
        list: List of document chunks if successful, None otherwise.
    """
    raw_text = load_document(file_path)
    if not raw_text:
        print_with_date("❌ No se pudo cargar el archivo de documentación.")
        return None


    chunks = parse_document(raw_text)


    if not chunks:
        print_with_date("❌ No se encontraron preguntas válidas en el documento.")
        return None

    print_with_date("✅ Documento procesado exitosamente.")
    print_with_date(f"Se extrajeron {len(chunks)} bloques de información del documento.")
    return chunks


@st.cache_resource
def setup_vector_store(chunks, selected_model):
    """
    Sets up the vector store using embeddings from the selected model.

    Args:
        chunks (list): List of document chunks.
        selected_model (str): Selected model name.

    Returns:
        FAISS: Vector store object if successful, None otherwise.
    """
    try:
        load_local_embbedings = os.getenv("LOAD_LOCAL_EMBBEDINGS", "No")
        print_with_date(f"Load local embbedings: {load_local_embbedings}" )
        vs_file = os.getenv("VECTOR_STORE_FILE", "No file was found")
        embeddings = OllamaEmbeddings(model=selected_model)
        if load_local_embbedings == "No":
            print_with_date(f"Creando embeddings con el modelo {selected_model}")
            mini_chunks = []
            chunk_size = int(os.getenv("MAX_CHUNK_SIZE", 300))
            for m_ch in chunks:
                tot_chunks = int(len(m_ch)/300)
                for i in range(tot_chunks):
                    if i == 0:
                        ch = m_ch[:chunk_size]
                    elif i == tot_chunks - 1:
                        ch = m_ch[i * 300 - 100:]
                    else:
                        ch = m_ch[i * 300 - 100: (i+1) * 300]
                    mini_chunks.append(ch)
            print_with_date(f"A total of {len(mini_chunks)} was identified.")
            vector_store = FAISS.from_texts(mini_chunks, embeddings, distance_strategy='cosine')
            vector_store.save_local(vs_file)
            print_with_date(f"Vector store was saved in the file {vs_file}")
        else:
            # FAISS.allow_dangerous_deserialization = True
            vector_store = FAISS.load_local(vs_file, embeddings)
            print_with_date(f"Vector store was loaded from the file {vs_file}")
        print_with_date("✅ Vector Store configurado exitosamente.")
        return vector_store
    except Exception as e:
        print_with_date(f"❌ Error al configurar el vector store: {e}")
        return None


@st.cache_resource
def initialize_qa_chain(_vector_store, selected_model):
    """
    Initializes the QA chain with the vector store and selected model.

    Args:
        _vector_store (FAISS): Vector store object. The underscore tells Streamlit not to hash this argument.
        selected_model (str): Selected model name.

    Returns:
        RetrievalQA: QA chain object if successful, None otherwise.
    """
    try:
        llm = create_model(selected_model)
        k = os.getenv("k", 3)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=_vector_store.as_retriever(search_kwargs={"k": k}),
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template=PROMPT_TEMPLATE,
                    input_variables=["context", "question"]
                )
            }
        )
        print_with_date("✅ QA Chain inicializada exitosamente.")
        return qa_chain
    except Exception as e:
        print_with_date(f"❌ Error al inicializar la cadena QA: {e}")
        return None
