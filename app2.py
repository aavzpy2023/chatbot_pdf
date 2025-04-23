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

def is_valid_json(json_data):
    try:
        # Intenta convertir el JSON a un objeto Python
        json.loads(json_data)
        return True
    except ValueError as e:
        # Captura errores de formato JSON
        print(f"El JSON no es válido: {e}")
        return False

#############################################
# Función para obtener la respuesta de Qwen2.5:3b vía API
#############################################
def get_answer_from_qwen(prompt: str, model: str = "qwen2.5:3B", endpoint: str = "http://localhost:5000/get_answer/"):
    # Validar entradas
    if not prompt or not isinstance(prompt, str):
        return "Error: El parámetro 'prompt' debe ser una cadena no vacía."
    if not model or not isinstance(model, str):
        return "Error: El parámetro 'model' debe ser una cadena no vacía."

    # Construir el payload
    payload = {"model": model, "prompt": prompt, "stream": False, "port": 11434}

    # Validar el JSON
    try:
        json_payload = json.dumps(payload)
        json.loads(json_payload)  # Verificar que el JSON sea válido
    except ValueError as e:
        return f"Error: El JSON generado no es válido: {e}"

    # Imprimir el payload para depuración
    print("Payload enviado:", json_payload)

    # Enviar la solicitud
    try:
        response = requests.post(endpoint, json=payload, timeout=120)
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


def validate_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Verificar que todos los bloques tengan un ID y contenido
    pattern = r"ID:\s*[A-Za-z0-9_-]+_[\w-]+_P\d+"
    ids = re.findall(pattern, content)
    if not ids:
        raise ValueError("El archivo no contiene IDs válidos.")

    print(f"IDs encontrados: {ids}")
    return ids


def get_question_contents(questions_id: list):
    file_path = "./documents/mf3.txt"
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Expresión regular para capturar IDs y su contenido
        pattern = r"(ID:\s*[A-Za-z0-9_-]+_[\w-]+_P\d+)\s*([\s\S]*?)(?=ID:\s*[A-Za-z0-9_-]+_[\w-]+_P\d+|\Z)"
        matches = re.findall(pattern, content)

        # Crear un diccionario con los IDs como claves y el contenido como valores
        result = {}
        for match in matches:
            question_id = match[0].strip()
            question_content = match[1].strip()
            print(f"ID: {question_id}, Content Length: {len(question_content)}")
            result[question_id] = question_content

        # Ajustar los IDs en questions_id para que coincidan con las claves del diccionario
        adjusted_questions_id = [f"ID: {c_id}" for c_id in questions_id]

        # Obtener los contenidos correspondientes a los IDs ajustados
        contents = [
            result.get(c_id, f"ID no encontrado: {c_id}") for c_id in adjusted_questions_id
        ]

        return contents

    except Exception as e:
        st.error(f"Error al leer el archivo o extraer los datos: {e}")
        return []


#############################################
# Función principal de la app
#############################################
def main():
    # Configurar la página en Streamlit
    st.set_page_config(
        page_title="Asistente Virtual Versat", page_icon="📚", layout="wide"
    )
    st.title("📚 Asistente Virtual Versat")
    st.markdown("Bienvenido al Asistente Virtual Versat.")

    # Conectar a Milvus
    client = MilvusClient(uri="http://localhost:19530")
    client.using_database("versat")

    # Cargar la colección "sarasola"
    collection_name = "sarasola"
    if not client.has_collection(collection_name):
        st.error(f"La colección '{collection_name}' no existe en la base de datos.")
        st.stop()

    client.load_collection(collection_name)

    # Entrada de la pregunta del usuario
    user_query = st.chat_input("Escribe tu pregunta aquí")
    if not user_query:
        st.stop()

    with st.spinner("Pensando..."):
        # Obtener el embedding de la pregunta
        vt_search = get_embedding_ollama(user_query)
        if not vt_search or not isinstance(vt_search, list):
            st.error("El proceso de embeddings ha fallado o devolvió datos inválidos.")
            st.stop()

        # Buscar en Milvus
        try:
            res_query = client.search(
                collection_name="sarasola",
                anns_field="q_vector",
                data=[vt_search],
                limit=2,
                search_params={"metric_type": "COSINE"},
            )[0]
            id_interest = [item.get("id") for item in res_query]
            if not id_interest:
                st.warning("No se encontraron IDs válidos para buscar en el archivo.")
                st.stop()
            id_interest = list(set(id_interest))
            resultados = "\n\n".join(get_question_contents(id_interest))
        except Exception as e:
            st.error(f"Error durante la búsqueda: {e}")
            st.stop()

        # Construir el prompt
        prompt = f"""
        Utiliza el siguiente contexto para responder la pregunta del usuario de forma clara y precisa. Brinda la mayor cantidad de detalles al usuario de forma tal que aclare su duda.

        Contexto:
        {resultados}

        Pregunta: {user_query}

        Respuesta:
        """

        # Mostrar mensaje de progreso
        st.info("Construyendo respuesta final con qwen2.5:3b...")

        # Obtener la respuesta del modelo
        answer = get_answer_from_qwen(model="qwen2.5:3B", prompt=prompt)

        # Mostrar la respuesta
        st.subheader("Respuesta Generada:")
        st.write(answer)


if __name__ == "__main__":
    main()
