import json
import logging
import os
import re
import requests
import streamlit as st
from pymilvus import MilvusClient
import datetime
from typing import List, Dict, Optional, Tuple, Any

# --- Constantes y Configuración ---
MILVUS_URI = "http://localhost:19530" # Cambiado a URI según pymilvus v2.3+
MILVUS_DB_NAME = "versat"
MILVUS_COLLECTION_NAME = "sarasola"
EMBEDDING_API_URL = "http://localhost:5000/generate-embeddings/"
LLM_API_URL = "http://localhost:5000/get_answer/"
CONTENT_FILE_PATH = "./documents/mf3.txt"
TARGET_UNIQUE_CONTEXT_COUNT = 2 # Número de documentos únicos distintos a recuperar
MILVUS_SEARCH_LIMIT = 10 # Cuántos resultados iniciales pedir a Milvus (debe ser >= TARGET_UNIQUE_CONTEXT_COUNT)
REQUEST_TIMEOUT = 60 # Timeout para llamadas a API (en segundos)
LLM_REQUEST_TIMEOUT = 500 # Timeout más largo para la llamada al LLM

# Configuración del Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Funciones Auxiliares ---

def print_with_date(message: str):
    """Imprime un mensaje con la fecha y hora actual y lo logea."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"{now} - {message}"
    print(log_message) # Para visibilidad en consola si se ejecuta localmente
    logging.info(message) # Para el logger configurado

@st.cache_data # Cachear la lectura del archivo si no cambia
def load_document_content(file_path: str) -> Dict[str, str]:
    """
    Carga el contenido del archivo y lo parsea en un diccionario ID -> Contenido.
    La clave será el ID completo (ej: "ID: Conf_Cap-III_P64").
    El valor será el texto completo del bloque, incluyendo la línea del ID.
    """
    content_map = {}
    try:
        print_with_date(f"Intentando cargar y parsear el archivo: {file_path}")
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Expresión regular para capturar ID y su contenido hasta el próximo ID o fin de archivo
        pattern = r"^(ID:\s*[A-Za-z0-9_-]+_[\w-]+_P\d+)\s*([\s\S]*?)(?=^ID:\s*[A-Za-z0-9_-]+_[\w-]+_P\d+|\Z)"
        matches = re.finditer(pattern, content, re.MULTILINE)

        count = 0
        for match in matches:
            full_id_line = match.group(1).strip() # "ID: Conf_Cap-III_P64"
            question_content = match.group(2).strip()
            # Almacenar usando la línea completa del ID como clave
            content_map[full_id_line] = f"{full_id_line}\n{question_content}"
            count += 1

        print_with_date(f"Parseado exitoso. Se encontraron {count} bloques de ID en '{os.path.basename(file_path)}'.")

        if not content_map:
            logging.warning(f"No se encontraron bloques de ID válidos en el archivo: {file_path}")
            st.warning(f"Advertencia: No se encontraron datos estructurados en '{os.path.basename(file_path)}'.")

    except FileNotFoundError:
        logging.error(f"Error Crítico: El archivo de contenido no se encontró en {file_path}")
        st.error(f"Error crítico: No se pudo encontrar el archivo de datos '{os.path.basename(file_path)}'. La aplicación no puede funcionar.")
        content_map = {} # Devolver diccionario vacío en caso de error fatal
    except Exception as e:
        logging.error(f"Error inesperado al leer o parsear el archivo {file_path}: {e}", exc_info=True)
        st.error(f"Error inesperado al procesar el archivo de datos: {e}")
        content_map = {}

    return content_map

@st.cache_resource # Cachear el cliente Milvus para reutilizar la conexión
def get_milvus_client() -> Optional[MilvusClient]:
    """Inicializa y devuelve un cliente Milvus conectado y con la colección cargada."""
    try:
        print_with_date(f"Intentando conectar a Milvus en {MILVUS_URI}...")
        client = MilvusClient(uri=MILVUS_URI)
        print_with_date(f"Conectado a Milvus. Intentando usar base de datos '{MILVUS_DB_NAME}'...")
        client.using_database(MILVUS_DB_NAME)
        print_with_date(f"Base de datos '{MILVUS_DB_NAME}' seleccionada.")

        print_with_date(f"Verificando existencia de la colección '{MILVUS_COLLECTION_NAME}'...")
        if not client.has_collection(MILVUS_COLLECTION_NAME):
            st.error(f"Error crítico: La colección '{MILVUS_COLLECTION_NAME}' no existe en la base de datos '{MILVUS_DB_NAME}'.")
            logging.error(f"La colección '{MILVUS_COLLECTION_NAME}' no existe en la DB '{MILVUS_DB_NAME}'.")
            return None

        print_with_date(f"Colección encontrada. Intentando cargar '{MILVUS_COLLECTION_NAME}' en memoria...")
        client.load_collection(MILVUS_COLLECTION_NAME)
        print_with_date(f"Colección '{MILVUS_COLLECTION_NAME}' cargada exitosamente.")
        return client
    except Exception as e:
        st.error(f"No se pudo conectar o preparar la colección en Milvus: {e}")
        logging.error(f"Error conectando/cargando colección Milvus: {e}", exc_info=True)
        return None

def get_embedding_from_api(text: str) -> Optional[List[float]]:
    """Obtiene el embedding para un texto desde la API Flask."""
    if not text:
        logging.warning("Se intentó obtener embedding para texto vacío.")
        return None
    payload = {"texts": [text]}
    print_with_date(f"Enviando texto a API de embeddings: {EMBEDDING_API_URL}")
    try:
        response = requests.post(EMBEDDING_API_URL, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status() # Lanza excepción para errores HTTP 4xx/5xx
        data = response.json()
        # Validar estructura de respuesta: {"embeddings": [[[vector_float]]]}
        if ("embeddings" in data and isinstance(data["embeddings"], list) and
                len(data["embeddings"]) > 0 and isinstance(data["embeddings"][0], list) and
                len(data["embeddings"][0]) > 0 and isinstance(data["embeddings"][0][0], list) and
                all(isinstance(val, (float, int)) for val in data["embeddings"][0][0])):
            embedding_vector = data["embeddings"][0][0]
            print_with_date(f"Embedding recibido exitosamente (Dimensión: {len(embedding_vector)}).")
            return embedding_vector
        else:
            logging.error(f"Formato inesperado de embeddings recibido: {data}")
            st.error("Error: La API de embeddings devolvió un formato de datos inesperado.")
            return None
    except requests.exceptions.Timeout:
        logging.error(f"Timeout al conectar con API de embeddings: {EMBEDDING_API_URL}")
        st.error("Error: La solicitud de embedding tardó demasiado en responder.")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error de conexión al llamar a la API de embeddings: {e}")
        st.error(f"Error de conexión al obtener embeddings: {e}")
        return None
    except json.JSONDecodeError:
        logging.error(f"No se pudo decodificar JSON de la API de embeddings. Respuesta: {response.text}")
        st.error("Error: Respuesta inválida (no JSON) recibida de la API de embeddings.")
        return None
    except Exception as e:
        logging.error(f"Error inesperado en get_embedding_from_api: {e}", exc_info=True)
        st.error("Ocurrió un error inesperado durante el proceso de embedding.")
        return None

def search_milvus(client: MilvusClient, vector: List[float], top_k: int) -> List[Dict[str, Any]]:
    """Realiza la búsqueda en Milvus y devuelve los resultados como lista de dicts."""
    if not client:
        st.error("Intento de búsqueda con cliente Milvus no inicializado.")
        logging.error("search_milvus llamado sin cliente válido.")
        return []
    if not vector:
        st.error("Intento de búsqueda sin un vector válido.")
        logging.error("search_milvus llamado sin vector.")
        return []

    print_with_date(f"Iniciando búsqueda en Milvus (Top K={top_k})...")
    try:
        search_params = {"metric_type": "COSINE", "params": {}}

        results = client.search(
            collection_name=MILVUS_COLLECTION_NAME,
            data=[vector],               # El vector de búsqueda debe estar en una lista
            anns_field="q_vector",       # Nombre del campo vectorial en tu esquema Milvus
            param=search_params,         # Parámetros de búsqueda
            limit=top_k,                 # Número de resultados a devolver
            output_fields=["id"]         # Campos adicionales a devolver (además de distancia/score)
        )

        if results and len(results) > 0:
            hits_data = []
            for hit in results[0]:
                distance = getattr(hit, 'distance', getattr(hit, 'score', None))
                if hit.id is not None and distance is not None:
                     hits_data.append({"id": hit.id, "distance": distance})
                else:
                     logging.warning(f"Resultado de Milvus incompleto: {hit}")

            print_with_date(f"Milvus devolvió {len(hits_data)} resultados válidos.")
            hits_data.sort(key=lambda x: x['distance']) # Asumiendo métrica de distancia (menor es mejor)
            return hits_data
        else:
            print_with_date("Milvus no devolvió resultados para la búsqueda.")
            return []
    except Exception as e:
        logging.error(f"Error durante la búsqueda en Milvus: {e}", exc_info=True)
        st.error(f"Error al buscar en la base de datos vectorial: {e}")
        return []

def get_top_unique_content(search_results: List[Dict[str, Any]], content_map: Dict[str, str], count: int) -> str:
    """
    Selecciona los IDs únicos más relevantes según el orden de `search_results`
    y recupera su contenido completo desde `content_map`.
    """
    top_unique_ids_simple = []
    seen_ids_simple = set()
    context_texts = []

    print_with_date(f"Procesando {len(search_results)} resultados de Milvus para obtener los {count} más relevantes únicos...")

    if not content_map:
        logging.error("El mapa de contenido está vacío. No se puede recuperar contexto.")
        st.error("Error interno: La base de conocimiento no está cargada.")
        return ""

    for hit in search_results:
        simple_doc_id = hit.get("id") # ID sin el prefijo "ID: ", ej: "Conf_Cap-III_P64"
        distance = hit.get("distance", "N/A")

        if simple_doc_id is None:
            logging.warning(f"Resultado de búsqueda Milvus sin ID simple: {hit}")
            continue

        if simple_doc_id not in seen_ids_simple:
            full_doc_id_key = f"ID: {simple_doc_id}"
            content = content_map.get(full_doc_id_key)

            if content:
                print_with_date(f"  -> ID Único encontrado: {simple_doc_id} (Distancia: {distance:.4f}). Añadiendo contexto.")
                context_texts.append(content)
                seen_ids_simple.add(simple_doc_id)
                top_unique_ids_simple.append(simple_doc_id)

                if len(context_texts) == count:
                    print_with_date(f"     Alcanzado el objetivo de {count} contextos únicos.")
                    break
            else:
                logging.warning(f"Contenido no encontrado en content_map para la clave: '{full_doc_id_key}' (ID Milvus: {simple_doc_id}).")
                print_with_date(f"  -> ¡Advertencia! Contenido no encontrado para ID: {simple_doc_id} (Clave: {full_doc_id_key})")
        else:
             print_with_date(f"  -> ID Repetido: {simple_doc_id} (Distancia: {distance:.4f}). Ignorando.")

    if not context_texts:
        print_with_date("No se pudo recuperar contenido para ningún ID único relevante.")
        st.warning("No se encontró contenido detallado para los resultados más relevantes.")
        return "" # Devolver string vacío si no hay contexto

    separator = "\n\n---\n[Documento Relevante Siguiente]\n---\n\n"
    final_context = separator.join(context_texts)
    print_with_date(f"Contexto final construido con {len(context_texts)} documento(s) único(s).")
    return final_context

def get_llm_response(prompt: str, model: str) -> str:
    """Obtiene la respuesta del LLM desde la API Flask."""
    if not prompt:
        return "Error interno: Se intentó llamar al LLM con un prompt vacío."
    payload = {"model": model, "prompt": prompt, "stream": False, "port": 11434}
    print_with_date(f"Enviando prompt al modelo LLM '{model}' via API: {LLM_API_URL}")
    try:
        response = requests.post(LLM_API_URL, json=payload, timeout=LLM_REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        if "response" in data and isinstance(data["response"], str):
            print_with_date("Respuesta recibida exitosamente del LLM.")
            return data["response"].strip()
        else:
            logging.error(f"Respuesta inesperada de la API LLM: {data}")
            return "Error: La API del modelo de lenguaje devolvió una respuesta inesperada o vacía."
    except requests.exceptions.Timeout:
        logging.error(f"Timeout ({LLM_REQUEST_TIMEOUT}s) esperando respuesta de la API LLM.")
        return f"Error: El modelo de lenguaje ('{model}') tardó demasiado en responder (más de {LLM_REQUEST_TIMEOUT} segundos)."
    except requests.exceptions.RequestException as e:
        logging.error(f"Error de conexión con la API LLM: {e}")
        return f"Error de conexión al contactar el modelo de lenguaje ('{model}'): {e}"
    except json.JSONDecodeError:
        logging.error(f"No se pudo decodificar la respuesta JSON de la API LLM: {response.text}")
        return "Error: Respuesta inválida (no JSON) recibida del modelo de lenguaje."
    except Exception as e:
        logging.error(f"Error inesperado en get_llm_response: {e}", exc_info=True)
        return f"Error inesperado al comunicarse con el modelo de lenguaje ('{model}')."

def build_final_prompt(context: str, user_query: str) -> str:
    """Construye el prompt final para el LLM, incluyendo rol, instrucciones, contexto y pregunta."""
    prompt = f"""
    **Rol:** Eres un asistente experto especializado en el software Versat Sarasola. Tu única fuente de conocimiento es la documentación proporcionada en el "Contexto".

    **Tarea:** Analiza detenidamente el "Contexto", que contiene uno o más fragmentos relevantes de la documentación oficial. Responde la "Pregunta del Usuario" de manera precise, útil y concisa, basándote *estrictamente* en la información encontrada en el contexto.

    **Instrucciones Cruciales:**
    1.  **Prioriza la Información:** Al buscar la respuesta, enfócate principalmente en las secciones `Sct. Respuesta` y `Sct. Pasos a Seguir` dentro de cada fragmento del contexto.
    2.  **Sintetiza, No Copies:** Combina la información relevante de todos los fragmentos en una respuesta única y coherente. Evita simplemente listar o copiar secciones enteras. No incluyas información irrelevante como Palabras Claves, Prioridad, etc., a menos que sea directamente parte de la respuesta a la pregunta.
    3.  **Claridad y Concisión:** Formula una respuesta directa y fácil de entender. Evita saludos genéricos o frases introductorias innecesarias. Ve al grano.
    4.  **Estrictamente Basado en el Contexto:** Tu respuesta DEBE basarse *únicamente* en el texto proporcionado en el "Contexto". **NO inventes información**, no hagas suposiciones ni utilices conocimiento externo sobre Versat Sarasola o cualquier otro tema.
    5.  **Manejo de Información Faltante:** Si el contexto no contiene la información necesaria para responder la pregunta de forma completa y precisa, indícalo claramente. Por ejemplo: "Según la documentación proporcionada, se menciona X, pero no se detallan los pasos específicos para Y." o "La información sobre Z no se encuentra en los fragmentos de contexto proporcionados." No respondas si no tienes la información en el contexto.
    6.  **Formato:** Usa párrafos cortos y, si es aplicable, listas numeradas o con viñetas para mejorar la legibilidad.

    **Contexto:**
    ```text
    {context if context else "No se proporcionó contexto relevante de la documentación."}
    Pregunta del Usuario:
    {user_query}

    Respuesta (Basada únicamente en el Contexto):
    """
    return prompt



def configure_sidebar() -> Optional[str]:
    """Configura la barra lateral para seleccionar el modelo LLM y muestra información."""
    with st.sidebar:
        st.header("⚙️ Configuración del Asistente")
        selected_model = None
        try:
            import ollama
            models_info = ollama.list().get("models", [])
            available_models = sorted([model["name"] for model in models_info])

            if not available_models:
                st.warning("No se detectaron modelos de Ollama disponibles. Introduce el nombre manualmente.")
                selected_model = st.text_input("Nombre del Modelo LLM:", value="qwen2:1.5b", key="model_input_manual")
            else:
                default_model = "qwen2:1.5b"
                if default_model not in available_models:
                    default_model = available_models[0]

                try:
                    default_index = available_models.index(default_model)
                except ValueError:
                    default_index = 0 # Fallback si el default no está por alguna razón

                selected_model = st.selectbox(
                    "Selecciona Modelo LLM (Ollama):",
                    available_models,
                    index=default_index,
                    key="model_selection_dynamic",
                    help="Elige el modelo de lenguaje que procesará tus preguntas."
                )

        except ImportError:
            st.warning("La librería 'ollama' no está instalada. Introduce el nombre del modelo manualmente.", icon="⚠️")
            selected_model = st.text_input("Nombre del Modelo LLM:", value="qwen2:1.5b", key="model_input_ollama_missing")
        except Exception as e:
            st.error(f"Error al interactuar con Ollama: {e}", icon="🔥")
            logging.error(f"Error al listar modelos de Ollama: {e}", exc_info=True)
            selected_model = st.text_input("Nombre del Modelo LLM (Error al listar):", value="qwen2:1.5b", key="model_input_ollama_error")

    st.divider()
    st.markdown("### Acerca De")
    st.info("""
    **Asistente Virtual Versat Sarasola**

    Este asistente responde preguntas utilizando:
    - Búsqueda semántica en **Milvus**.
    - Un modelo de lenguaje local (LLM) vía **Ollama**.
    - Base de conocimiento del archivo `mf3.txt`.
    """)

    return selected_model.strip() if selected_model else None


def main():
    """Función principal que ejecuta la aplicación Streamlit."""
    st.set_page_config(
    page_title="Asistente Versat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
    )
    st.title("📚 Asistente Virtual Versat Sarasola")
    st.caption("Realiza tus consultas sobre el sistema. Las respuestas se basan en la documentación cargada.")

    # --- Inicialización y Configuración ---
    selected_llm_model = configure_sidebar()
    content_data_map = load_document_content(CONTENT_FILE_PATH)
    milvus_db_client = get_milvus_client()

    if not selected_llm_model:
        st.error("⛔ Por favor, selecciona o introduce un modelo LLM válido en la barra lateral para continuar.")
        st.stop()
    if not content_data_map:
         st.error("⛔ No se pudo cargar la base de conocimiento. El asistente no puede operar.")
         st.stop()
    if not milvus_db_client:
         st.error("⛔ No se pudo conectar a la base de datos vectorial (Milvus). El asistente no puede operar.")
         st.stop()

    print_with_date(f"Asistente listo. Modelo LLM seleccionado: {selected_llm_model}")

    # --- Gestión del Historial de Chat ---
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": f"¡Hola! Soy tu asistente para Versat Sarasola. Usaré el modelo '{selected_llm_model}'. ¿En qué puedo ayudarte?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Manejo de la Entrada del Usuario ---
    if user_query := st.chat_input("Escribe tu pregunta sobre Versat Sarasola aquí..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("Pensando... 🤔")

            final_llm_response = ""
            try:
                with st.status("Procesando tu pregunta...", expanded=False) as status:

                    # --- PASO 1: Obtener Embedding ---
                    status.update(label="🧠 Vectorizando pregunta...", state="running")
                    query_vector = get_embedding_from_api(user_query)
                    if not query_vector:
                        error_msg = "❌ No se pudo obtener la representación vectorial de tu pregunta. Intenta reformularla."
                        status.update(label=error_msg, state="error", expanded=True)
                        response_placeholder.error(error_msg)
                        final_llm_response = error_msg # Guardar error para historial
                        st.session_state.messages.append({"role": "assistant", "content": final_llm_response})
                        st.stop() # No continuar sin vector

                    # --- PASO 2: Buscar en Milvus ---
                    status.update(label=f"🔍 Buscando documentos relevantes en Milvus (Top {MILVUS_SEARCH_LIMIT})...", state="running")
                    milvus_results = search_milvus(milvus_db_client, query_vector, top_k=MILVUS_SEARCH_LIMIT)
                    retrieved_context = "" # Inicializar contexto vacío
                    if not milvus_results:
                        warning_msg = f"⚠️ No encontré documentos directamente relacionados en Milvus para tu consulta. Intentaré responder sin contexto específico de la documentación."
                        status.update(label=warning_msg, state="complete", expanded=False)
                        st.warning(warning_msg, icon="ℹ️")
                    else:
                        # --- PASO 3: Obtener Contexto Único Relevante ---
                        status.update(label=f"📚 Seleccionando los {TARGET_UNIQUE_CONTEXT_COUNT} documentos únicos más importantes...", state="running")
                        retrieved_context = get_top_unique_content(milvus_results, content_data_map, TARGET_UNIQUE_CONTEXT_COUNT)
                        if not retrieved_context:
                            warning_msg = f"ℹ️ Encontré referencias en Milvus, pero no pude cargar el contenido detallado. Intentaré responder sin ese contexto."
                            status.update(label=warning_msg, state="complete", expanded=False)
                            st.warning(warning_msg, icon="ℹ️")

                    # --- PASO 4: Construir Prompt y Llamar al LLM ---
                    status.update(label=f"✍️ Generando respuesta con el modelo '{selected_llm_model}'...", state="running")
                    final_prompt_for_llm = build_final_prompt(retrieved_context, user_query)
                    final_llm_response = get_llm_response(final_prompt_for_llm, selected_llm_model)

                    response_placeholder.markdown(final_llm_response)
                    status.update(label="¡Respuesta generada!", state="complete", expanded=False)

            except Exception as e:
                logging.error(f"Error inesperado en el procesamiento de la consulta: {e}", exc_info=True)
                final_llm_response = f"🔴 Ocurrió un error inesperado al procesar tu pregunta: {e}"
                response_placeholder.error(final_llm_response)
                if 'status' in locals() and status:
                     status.update(label="Error inesperado", state="error", expanded=True)

            # Añadir respuesta final (o mensaje de error) al historial
            st.session_state.messages.append({"role": "assistant", "content": final_llm_response})

            # Re-ejecutar para asegurar que el historial se muestra actualizado y hacer scroll (mejor intento)
            st.rerun()


if __name__ == "__main__":
    main()
