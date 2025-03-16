from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

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

    data = {}
    for section in content.split("Sct."):
        if ":" in section:
            splits = section.split(":")
            data[splits[0].lower()] = splits[1]
        else:
            data["id"] = section
    # id_match = re.search(r'ID:\s*(.+)', content)
    # titulo_match = re.search(r'Título:\s*(.+)', content)
    # categoria_match = re.search(r'Categoría:\s*(.+)', content)
    # prioridad_match = re.search(r'Prioridad:\s*(.+)', content)
    # version_match = re.search(r'Versión:\s*(.+)', content)
    # fecha_match = re.search(r'Fecha de Actualización:\s*(.+)', content)
    # funcionalidad_match = re.search(r'Funcionalidad:\s*([\s\S]+?)(?=Respuesta:|Roles de Acceso:|$)', content)
    # respuesta_match = re.search(r'Respuesta:\s*([\s\S]+?)(?=Roles de Acceso:|Variantes de Preguntas:|Pasos a Seguir:|Notas Adicionales:|Errores Comunes y Soluciones:|Prerrequisitos:|Resultados Esperados:|Referencias Relacionadas:|Escenarios Posibles:|Casos Especiales:|$)', content)
    # roles_match = re.search(r'Roles de Acceso:\s*([\s\S]+?)(?=Variantes de Preguntas:|Pasos a Seguir:|Notas Adicionales:|Errores Comunes y Soluciones:|Prerrequisitos:|Resultados Esperados:|Referencias Relacionadas:|Escenarios Posibles:|Casos Especiales:|$)', content)
    # variantes_match = re.findall(r'¿(.*?)\?(.*?)(?=¿|\Z)', content, re.DOTALL)
    # pasos_match = re.search(r'Pasos a Seguir:\s*([\s\S]+?)(?=Notas Adicionales:|Errores Comunes y Soluciones:|Prerrequisitos:|Resultados Esperados:|Referencias Relacionadas:|Escenarios Posibles:|Casos Especiales:|$)', content)
    # errores_match = re.findall(r'"(.*?)":\s*(.*?)\n', content, re.DOTALL)
    # resultados_match = re.search(r'Resultados Esperados:\s*([\s\S]+?)(?=Referencias Relacionadas:|Escenarios Posibles:|Casos Especiales:|$)', content)
    # escenarios_match = re.findall(r'Escenarios Posibles:\s*([\s\S]+?)(?=Casos Especiales:|Feedback del Usuario:|$)', content, re.DOTALL)
    # casos_especiales_match = re.search(r'Casos Especiales:\s*([\s\S]+?)(?=Feedback del Usuario:|$)', content)
    # feedback_match = re.search(r'Feedback del Usuario:\s*([\s\S]+)', content)

    return data
    # {
        # "id": id_match.group(1).strip() if id_match else "No especificado.",
        # "titulo": titulo_match.group(1).strip() if titulo_match else "No especificado.",
        # "categoria": categoria_match.group(1).strip() if categoria_match else "No especificada.",
        # "prioridad": prioridad_match.group(1).strip() if prioridad_match else "No especificada.",
        # "version": version_match.group(1).strip() if version_match else "No especificada.",
        # "fecha": fecha_match.group(1).strip() if fecha_match else "No especificada.",
        # "funcionalidad": funcionalidad_match.group(1).strip() if funcionalidad_match else "No especificada.",
        # "respuesta": respuesta_match.group(1).strip() if respuesta_match else "No especificada.",
        # "roles": roles_match.group(1).strip() if roles_match else "No especificados.",
        # "variantes": [{"pregunta": v[0].strip(), "respuesta": v[1].strip()} for v in variantes_match] if variantes_match else [],
        # "pasos": [p.strip() for p in pasos_match.group(1).splitlines()] if pasos_match else [],
        # "errores": [(e[0].strip(), e[1].strip()) for e in errores_match] if errores_match else [],
        # "resultados": resultados_match.group(1).strip() if resultados_match else "No especificados.",
        # "escenarios": [e.strip() for e in escenarios_match] if escenarios_match else [],
        # "casos_especiales": casos_especiales_match.group(1).strip() if casos_especiales_match else "No especificados.",
        # "feedback": feedback_match.group(1).strip() if feedback_match else "No especificado."
    # }

# =============================================================================
# Función para dividir el documento en chunks basados en preguntas
# =============================================================================
def parse_document(text):
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

                # Construir el chunk final
                # chunk = f"""
                # ID: {detalles.get("id","")}
                # Pregunta: {detalles.get("pregunta","")}
                # Título: {detalles.get("titulo", "")}
                # Categoría: {detalles.get("categoria", "")}
                # Prioridad: {detalles.get("prioridad","")}
                # Versión: {detalles.get("version", "")}

                # Funcionalidad: {detalles.get("funcionalidad", "")}

                # Respuesta:
                # {detalles.get("respuesta", "")}

                # Roles de Acceso:
                # {detalles["roles"]}

                # Variantes de Preguntas:
                # {"; ".join([f"{v['pregunta']} -> {v['respuesta']}" for v in detalles["variantes"]])}

                # Pasos a Seguir:
                # {"; ".join(detalles["pasos"])}

                # Errores Comunes y Soluciones:
                # {"; ".join([f"{e[0]}: {e[1]}" for e in detalles["errores"]])}

                # Resultados Esperados:
                # {detalles["resultados"]}

                # Escenarios Posibles:
                # {", ".join(detalles["escenarios"])}

                # Casos Especiales:
                # {detalles["casos_especiales"]}

                # Feedback del Usuario:
                # {detalles["feedback"]}
                # """.strip()
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
        embeddings = OllamaEmbeddings(model=selected_model)
        vector_store = FAISS.from_texts(chunks, embeddings)
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
