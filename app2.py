import json
import os
import subprocess

import requests
import streamlit as st
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)


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
def get_answer_from_qwen(prompt: str, model: str = "qwen2.5:3b"):
    """
    Envía el prompt (contexto + pregunta) al endpoint de la API para obtener la respuesta del modelo.
    Se espera que la API acepte un JSON con las claves "model" y "prompt", y retorne un JSON con la clave "answer".
    """
    url = "http://localhost:5000/get_answer/"
    payload = {"model": model, "prompt": prompt}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["answer"]
    except Exception as e:
        return f"Error al obtener respuesta del modelo: {e}"


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
    st.markdown(
        "Bienvenido al Asistente Virtual Versat. Envía tu pregunta para buscar información en la base de datos vectorial y generar una respuesta final."
    )

    # 1. Conectar a Milvus (base de datos vectorial)
    connections.connect(alias="default", host="localhost", port="19530")
    st.info("Conexión a la base de datos vectorial establecida.")

    # 2. Seleccionar y cargar la colección "Sarasola"
    collection_name = "Sarasola"
    if not utility.has_collection(collection_name):
        st.error(f"La colección '{collection_name}' no existe en la base de datos.")
        st.stop()

    collection = Collection(name=collection_name)
    collection.load()
    st.success(f"Colección '{collection_name}' cargada correctamente.")

    # 3. (Opcional) Consulta de verificación para mostrar algunos datos
    try:
        query_result = collection.query(
            expr="my_id > 0", output_fields=["my_id", "my_vector", "text"], limit=5
        )
        st.write("Ejemplo de datos en la colección:")
        for record in query_result:
            st.write(record)
    except Exception as e:
        st.error(f"Error en la consulta de verificación: {e}")

    # 4. Entrada de la pregunta del usuario
    user_query = st.chat_input("Escribe tu pregunta aquí")
    if not user_query:
        st.info("Ingresa una pregunta para buscar en la base de datos vectorial.")
        st.stop()

    # 5. Convertir la pregunta a embedding usando la API de Ollama
    st.info("Convirtiendo la pregunta en embedding (nomic-embed-text vía API)...")
    vt_search = get_embedding_ollama(user_query)
    print("VT", vt_search)
    if vt_search is None:
        st.error("El proceso de embeddings ha fallado.")
        st.stop()

    st.write(
        f"Dimensiones del vector de búsqueda: {len(vt_search)}"
    )  # Se espera que sean 768

    # 6. Realizar la búsqueda en la colección
    st.info("Realizando búsqueda en la base de datos vectorial...")
    try:
        res_query = collection.search(
            data=[vt_search],  # vt_search debe ser una lista de 768 floats
            anns_field="my_vector",  # Campo vectorial según el esquema
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=3,  # Número de resultados a devolver
            output_fields=[
                "text",
                "my_id",
            ],  # Campos adicionales que servirán de contexto para el LLM
        )
    except Exception as e:
        st.error(f"Error durante la búsqueda: {e}")
        st.stop()

    # 7. Preparar y mostrar los resultados de la búsqueda
    resultados = ""
    for hits in res_query:
        for hit in hits:
            texto = hit.text if hasattr(hit, "text") else "Sin texto disponible"
            extra_id = hit.my_id if hasattr(hit, "my_id") else "Sin id"
            # resultados += (
            #     f"ID Interno: {hit.id} | Distancia: {hit.distance:.4f} | "
            #     f"my_id: {extra_id}\nTexto: {texto}\n\n"
            # )
            resultados += f"Texto: {texto}\n\n"

    if resultados:
        st.subheader("Resultados de la búsqueda:")
        st.text_area("Resultados", resultados, height=300)
    else:
        st.warning("No se encontraron resultados para esa pregunta.")

    # 8. Preparar el prompt para qwen2.5:3b usando el contexto y la pregunta original
    prompt = (
        "Utiliza el siguiente contexto para responder la pregunta del usuario de forma clara y precisa.\n\n"
        f"Contexto:\n{resultados}\n"
        f"Pregunta: {user_query}\n\nRespuesta:"
    )

    print("PROMPT", prompt)
    st.info("Construyendo respuesta final con qwen2.5:3b...")
    answer = get_answer_from_qwen(prompt=prompt[0])
    st.subheader("Respuesta Generada:")
    st.write(answer)


if __name__ == "__main__":
    main()
