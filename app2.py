import json
import logging
import os
import re
import subprocess

import requests
import streamlit as st
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
)

from milvus_search import search_vector

logging.basicConfig(level=logging.INFO)


#############################################
# Función para obtener embeddings vía API de Ollama
#############################################
def get_embedding_ollama(text: str):
    """
    Envía un POST a la API de embeddings de Ollama para convertir el texto en embedding.
    Se espera que la API acepte el payload {"texts": ["tu texto"]} y retorne un JSON con la clave "embeddings".
    """
    url = "http://localhost:5000/generate-embeddings/"
    payload = {"texts": [text]}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        # Se asume que la respuesta tiene el formato:
        # {"embeddings": [[0.123,...,0.456]]}
        return data["embeddings"][0][0]
    except Exception as e:
        st.error(f"Error al obtener el embedding desde la API: {e}")
        return None


#############################################
# Función para obtener la respuesta de Qwen2.5:3b vía API
#############################################
def get_answer_from_qwen(
    prompt: str,
    model: str = "qwen2.5:3b",
    endpoint: str = "http://localhost:5000/get_answer/",
):
    """
    Envía el prompt (contexto + pregunta) al endpoint de la API para obtener la respuesta del modelo.
    Se espera que la API acepte un JSON con las claves "model" y "prompt", y retorne un JSON con la clave "answer".
    """
    # Validar entradas
    if not prompt or not isinstance(prompt, str):
        return "Error: El parámetro 'prompt' debe ser una cadena no vacía."
    if not model or not isinstance(model, str):
        return "Error: El parámetro 'model' debe ser una cadena no vacía."

    # Preparar la solicitud
    url = endpoint
    payload = {"model": model, "prompt": prompt}
    headers = {"Content-Type": "application/json"}

    try:
        # Realizar la solicitud con timeout
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()

        # Procesar la respuesta
        data = response.json()
        if "response" in data:
            return data["response"]
        else:
            logging.error(f"Respuesta inesperada del servidor: {data}")
            return "Error: La respuesta del servidor no tiene el formato esperado."

    except requests.exceptions.RequestException as e:
        logging.error(f"Error de conexión: {e}")
        return f"Error de conexión: {e}"
    except KeyError:
        logging.error("Error: La respuesta del servidor no contiene la clave 'answer'.")
        return "Error: La respuesta del servidor no tiene el formato esperado."


def get_question_contents(questions_id: list):
    file_path = "./documents/mf3.txt"  # Leer el archivo
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Expresión regular para capturar IDs y su contenido
    pattern = r"(ID:\s*[A-Za-z0-9_-]+_[\w-]+_P\d+)(.*?)(?=ID:\s*[A-Za-z0-9_-]+_[\w-]+_P\d+|\Z)"

    # Encontrar todas las coincidencias
    matches = re.findall(pattern, content, re.DOTALL)

    # Crear un diccionario con los IDs como claves y el contenido como valores
    result = {}
    for match in matches:
        question_id = match[0].strip()  # ID de la pregunta (incluye "ID: ")
        question_content = match[1].strip()  # Contenido asociado al ID
        result[question_id] = question_content

    # Ajustar los IDs en questions_id para que coincidan con las claves del diccionario
    adjusted_questions_id = [f"ID: {c_id}" for c_id in questions_id]

    # Obtener los contenidos correspondientes a los IDs ajustados
    contents = [result.get(c_id, "") for c_id in adjusted_questions_id]

    return contents


#############################################
# Función principal de la app
#############################################
def main():
    # Configurar la página en Streamlit
    st.set_page_config(
        page_title="Asistente Virtual Versat",
        page_icon="📚",
        layout="wide",
    )
    st.title("📚 Asistente Virtual Versat")
    st.markdown("Bienvenido al Asistente Virtual Versat.")

    # 1. Conectar a Milvus (base de datos vectorial)
    client = MilvusClient(uri="http://localhost:19530")
    client.using_database("versat")  # Activar la base de datos
    # st.info("Conexión a la base de datos vectorial establecida.")

    # 2. Seleccionar y cargar la colección "sarasola"
    collection_name = "sarasola"
    if not client.has_collection(collection_name):
        st.error(f"La colección '{collection_name}' no existe en la base de datos.")
        st.stop()

    client.load_collection(collection_name)
    # st.success(f"Colección '{collection_name}' cargada correctamente.")

    # 3. (Opcional) Consulta de verificación para mostrar algunos datos
    # try:
    #     query_result = client.query(
    #         collection_name,
    #         expr="q_id > 0",
    #         output_fields=["q_id", "q_vector", "q_question"],
    #         limit=5,
    #     )
    #     st.write("Ejemplo de datos en la colección:")
    #     for record in query_result:
    #         st.write(record)
    # except Exception as e:
    #     st.error(f"Error en la consulta de verificación: {e}")

    # 4. Entrada de la pregunta del usuario
    user_query = st.chat_input("Escribe tu pregunta aquí")
    if not user_query:
        # st.info("Ingresa una pregunta para buscar en la base de datos vectorial.")
        st.stop()
    st.write(user_query)
    with st.spinner("Pensando..."):
        # 5. Convertir la pregunta a embedding usando la API de Ollama
        # st.info("Convirtiendo la pregunta en embedding (nomic-embed-text vía API)...")
        vt_search = get_embedding_ollama(user_query)
        print("VT", vt_search)
        if vt_search is None:
            st.error("El proceso de embeddings ha fallado.")
            st.stop()

        # st.write(
        #     f"Dimensiones del vector de búsqueda: {len(vt_search)}"
        # )  # Se espera que sean 768

        # 6. Realizar la búsqueda en la colección
        # st.info("Realizando búsqueda en la base de datos vectorial...")
        try:
            res_query = client.search(
                collection_name="sarasola",
                anns_field="q_vector",
                data=[vt_search],
                limit=2,
                search_params={"metric_type": "L2"},
            )[0]
            id_interest = [item.get("id") for item in res_query]
            resultados = " ".join(get_question_contents(id_interest))

            # res_query = client.search(
            #     collection_name=collection_name,
            #     data=[vt_search],  # vt_search debe ser una lista de 768 floats
            #     anns_field="my_vector",  # Campo vectorial según el esquema
            #     param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            #     limit=3,  # Número de resultados a devolver
            #     output_fields=[
            #         "text",
            #         "my_id",
            #     ],  # Campos adicionales que servirán de contexto para el LLM
            # )

        except Exception as e:
            st.error(f"Error durante la búsqueda: {e}")
            st.stop()

        # 7. Preparar y mostrar los resultados de la búsqueda
        # resultados = ""
        # for hits in res_query:
        #     for hit in hits:
        #         texto = hit["text"] if "text" in hit else "Sin texto disponible"
        #         extra_id = hit["my_id"] if "my_id" in hit else "Sin id"
        #         resultados += f"Texto: {texto}\n\n"

        if resultados:
            # st.subheader("Resultados de la búsqueda:")
            # st.text_area("Resultados", resultados, height=300)
            # st.info("Construyendo respuesta final con qwen2.5:3b...")
            pass
        else:
            st.warning("No se encontraron resultados para esa pregunta.")

        # 8. Preparar el prompt para qwen2.5:3b usando el contexto y la pregunta original

        prompt = (
            "Utiliza el siguiente contexto para responder la pregunta del usuario de forma clara y precisa. Brinda la mayor cantidad de detalles al usuario de forma tal que aclare su duda\n\n"
            f"Contexto:\n{resultados}\n"
            f"Pregunta: {user_query}\n\nRespuesta:"
        )

        # print("PROMPT", prompt)
        # st.info("Construyendo respuesta final con qwen2.5:3b...")
        answer = get_answer_from_qwen(prompt=prompt)
        # st.subheader("Respuesta Generada:")
        st.write(answer)


if __name__ == "__main__":
    main()
