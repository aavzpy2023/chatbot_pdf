import json
import re

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

    sections = re.split(r"ID: ", content)[1:]
    result = {}

    for section in sections:
        id_match = re.match(r"(\S+)", section)
        if id_match:
            id_value = id_match.group(1)
            section_data = process_section(section)
            result[id_value] = section_data

    return result


def get_questions(json_data: dict):
    all_questions = []
    for id in json_data.keys():
        full_question_content: dict = json_data.get(id, "")
        question_question = full_question_content.get("Pregunta")
        if question_question:
            all_questions.append([id, question_question])
        q_variants: list = full_question_content.get("Variantes de Preguntas", "")
        all_questions.extend([[id, q_var[q_var.find("¿") :]] for q_var in q_variants])
    return all_questions


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


def create_database(client, db_name: str):
    print(f"Creating database {db_name}")
    if db_name not in client.list_databases():
        client.create_database(
            db_name=db_name, properties={"database.replica.number": 3}
        )
    else:
        print("Database already exists")

    # Activar la base de datos
    client.using_database(db_name)
    return client


def create_schema(client, collection_name: str):
    print("Creating schema")
    # Eliminar colección si existe
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    # Crear esquema
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("q_id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("q_vector", DataType.FLOAT_VECTOR, dim=768)
    schema.add_field("q_question", DataType.VARCHAR, max_length=512)

    # Crear colección
    client.create_collection(collection_name=collection_name, schema=schema)
    return client


def create_index(client, index_name: str, collection_name: str):
    print("Creating index")
    index_params = [
        {
            "field_name": index_name,
            "index_type": "IVF_SQ8",
            "metric_type": "COSINE",
            "params": {"nlist": 256},
        }
    ]
    try:
        client.create_index(collection_name=collection_name, index_params=index_params)
        print("Índice creado exitosamente.")
    except Exception as ex:
        print("Error al crear el índice:", ex)
    return client


if __name__ == "__main__":
    file_path = "./documents/mf3.txt"
    json_data = convert_text_to_json(file_path)
    answers = get_questions(json_data)
    ps = [p[1] for p in answers]
    emb = get_embeddings(ps)

    # Inicializar cliente de Milvus
    client = MilvusClient(uri="http://localhost:19530")

    # Crear y activar la base de datos "versat"
    client = create_database(client, db_name="versat")

    # Crear colección "sarasola" en la base de datos "versat"
    client = create_schema(client, collection_name="sarasola")

    # Crear índice en "sarasola"
    client = create_index(client, index_name="q_vector", collection_name="sarasola")

    # Aplanar los vectores si están anidados
    emb = [e[0] if isinstance(e, list) and len(e) == 1 else e for e in emb]

    # Inspeccionar los vectores generados
    print("Primeros 3 vectores generados:")
    for i, e in enumerate(emb[:3]):
        print(f"Vector {i + 1}: Longitud = {len(e)}, Datos = {e[:10]}...")

    # Preparar datos
    dt_ok = []
    for p, e in zip(answers, emb):
        if p[1]:  # Asegurar que q_question no esté vacío
            vector = e  # Extraer el vector
            if len(vector) == 768:  # Validar la longitud del vector
                dt_ok.append({"q_id": p[0], "q_vector": vector, "q_question": p[1]})
            else:
                print(f"Vector inválido para ID {p[0]}: Longitud = {len(vector)}")

    # Insertar datos
    print("Inserting data")
    res = client.insert(collection_name="sarasola", data=dt_ok)
    print("Inserción exitosa:", res)
