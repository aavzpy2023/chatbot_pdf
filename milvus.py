import json
import re

import numpy as np
import requests
from pymilvus import DataType, MilvusClient


def process_section(section):
    lines = section.strip().split("\n")
    data = {}
    current_key = None

    for line in lines:
        line = line.strip()
        if not line:
            continue  # Ignorar líneas en blanco
        if line.startswith("Sct."):
            # Extraer la clave y el valor inicial
            key = line[4:].split(":", 1)[0].strip()
            value = line[4:].split(":", 1)[1].strip() if ":" in line[4:] else ""
            if key in [
                "Variantes de Preguntas",
                "Pasos a Seguir",
                "Resultados Esperados",
                "Escenarios Posibles y Casos Especiales",
            ]:
                current_key = key
                data[key] = [value] if value else []
            else:
                data[key] = value
        elif current_key:
            data[current_key].append(line)

    return data


def convert_text_to_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    sections = re.split(r"ID: ", content)[1:-1]

    # print("SECTION:", list(sections)[-1])
    result = {}

    for section in sections:
        id_match = re.match(r"(\S+)", section)
        if id_match:
            id_value = id_match.group(1)
            section_data = process_section(section)
            result[id_value] = section_data

    return result


def get_answers(json_data: dict):
    pregs = [[id, json_data.get(id).get("Pregunta")] for id in json_data.keys()]
    return pregs


# URL de la API local
API_URL = "http://localhost:5000/generate-embeddings/"


def get_embeddings(texts):
    headers = {"Content-Type": "application/json"}
    data = {"texts": texts}
    response = requests.post(API_URL, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()["embeddings"]
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")


def create_collection(collection_name: str):
    print(f"Creating collection {collection_name}")
    client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
    if client.has_collection(collection_name="Versat"):
        client.drop_collection(collection_name="Versat")
        client.create_collection(
            collection_name="Versat",
            dimension=768,  # The vectors we will use in this demo has 768 dimensions
        )
    return client


def create_database(client, db_name: str):

    print(f"Creating database {db_name}")
    if "Versat" not in client.list_databases():
        client.create_database(
            db_name="Versat", properties={"database.replica.number": 3}
        )
    else:
        print("Database already exists")
    return client


def create_schema(client):
    print("Creating schema")
    schema = MilvusClient.create_schema(
        auto_id=False,  # No usamos ID automático
        enable_dynamic_field=False,  # Deshabilitar campos dinámicos
    )

    # Agregar campos al esquema
    schema.add_field(
        field_name="q_id", datatype=DataType.VARCHAR, is_primary=True, max_length=64
    )
    schema.add_field(
        field_name="q_vector", datatype=DataType.FLOAT_VECTOR, dim=768
    )  # Dimensión del vector
    schema.add_field(field_name="q_question", datatype=DataType.VARCHAR, max_length=512)

    # Crear la colección
    client.create_collection(collection_name="Sarasola", schema=schema)
    return client


def create_index(
    client,
    index_name: str,
):
    print("Creating index")
    index_params = [
        {
            "field_name": f"{index_name}",  # Campo al que se aplicará el índice
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        }
    ]
    # Crear el índice
    try:
        client.create_index(
            collection_name="Sarasola", index_params=index_params  # Ahora es una lista
        )
        print("Índice creado exitosamente.")
    except Exception as ex:
        print("Error al crear el índice:", ex)
    return client


if __name__ == "__main__":
    file_path = "./documents/mf3.txt"
    json_data = convert_text_to_json(file_path)
    answers = get_answers(json_data)
    ps = [f"{p[1]}" for p in answers]
    emb = get_embeddings(ps)
    client = create_collection("Versat")
    client = create_database(client=client, db_name="Sarasola")
    client = create_schema(client)
    client = create_index(client, index_name="q_vector")
    dt_ok = [
        {"q_id": p[0], "q_vector": e[0], "q_question": p[1]}
        for p, e in zip(answers, emb)
    ]
    print("inserting data")
    res = client.insert(collection_name="Sarasola", data=dt_ok)
