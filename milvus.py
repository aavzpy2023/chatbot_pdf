import random

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

# Importamos la librería para convertir la pregunta del usuario en embeddings
from sentence_transformers import SentenceTransformer

# 1. Conectar al servidor Milvus
connections.connect(alias="default", host="localhost", port="19530")
print("Conexión a Milvus establecida.")

# 2. Definir el esquema de la colección
fields = [
    FieldSchema(name="my_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="my_vector", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
]
schema = CollectionSchema(
    fields=fields, description="Colección de preguntas y respuestas"
)

# 3. Crear o cargar la colección
collection_name = "Sarasola"
if not utility.has_collection(collection_name):
    collection = Collection(name=collection_name, schema=schema)
    print(f"Colección '{collection_name}' creada exitosamente.")
else:
    collection = Collection(name=collection_name)
    print(f"Colección '{collection_name}' ya existe.")


# limpiar coleccion
# Eliminar todos los datos en la colección
# collection.delete(expr="")  # Esto elimina todos los registros de la colección.

# 4. Insertar datos en la colección
# Por ejemplo, definimos algunos datos de ejemplo.
# Cada registro es una lista: [my_id, my_vector, text]
dummy_vector = [random.uniform(-1, 1) for _ in range(768)]  # Vector de ejemplo (dummy)

data_rows = [
    [1, dummy_vector, "Texto de ejemplo para la pregunta 1"],
    [2, dummy_vector, "Texto de ejemplo para la pregunta 2"],
    # Puedes agregar más registros según lo necesites (hasta 18,402 u otra cantidad)
]

# Convertir de formato fila (row-based) a formato columnar (column-based)
data_columnar = list(zip(*data_rows))  # Esto genera una tupla de 3 elementos
data_columnar = [list(col) for col in data_columnar]

# Insertar los datos
insert_result = collection.insert(data_columnar)
print("Datos insertados exitosamente.")
print("IDs asignados:", insert_result.primary_keys)

# Hacer un flush para confirmar que los datos se han escrito
collection.flush()

# 5. Crear un índice sobre el campo vectorial "my_vector"
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",
    "params": {"nlist": 128},
}
collection.create_index(field_name="my_vector", index_params=index_params)
print("Índice creado en el campo 'my_vector'.")

# 6. Cargar la colección en memoria (necesario para consultas y búsquedas)
collection.load()
print("Colección cargada en memoria.")

# 7. Realizar una consulta para verificar los datos insertados
query_result = collection.query(
    expr="my_id > 0", output_fields=["my_id", "my_vector", "text"], limit=5
)
print("\nResultados de la consulta:")
for record in query_result:
    print(record)

# 8. Convertir la pregunta del usuario en embeddings para usarlos en la búsqueda
# Usamos SentenceTransformers con un modelo que devuelve 768 dimensiones, por ejemplo "all-mpnet-base-v2"
embedding_model = SentenceTransformer("all-mpnet-base-v2")
user_question = input("\nIngresa tu pregunta: ")
# Convertir la pregunta a un vector de embeddings (asegúrate que la dimensión sea 768)
vt_search = embedding_model.encode(user_question).tolist()
print("Dimensiones del vector de búsqueda:", len(vt_search))

# 9. Realizar una búsqueda de similitud que también retorne el campo "text" y "my_id"
res_query = collection.search(
    data=[vt_search],  # Vector de consulta (debe ser una lista de 768 floats)
    anns_field="my_vector",  # Campo de vectores, debe coincidir con el esquema
    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
    limit=3,  # Número de resultados a devolver
    output_fields=["text", "my_id"],  # Campos adicionales a retornar
)

print("\nResultados de la búsqueda:")
for hits in res_query:
    for hit in hits:
        # En PyMilvus, los campos adicionales se devuelven como atributos del hit
        texto = hit.text if hasattr(hit, "text") else "Ningún texto"
        my_id = hit.my_id if hasattr(hit, "my_id") else "Ningún id"
        print(
            f"ID Interno: {hit.id}, Distancia: {hit.distance}, my_id: {my_id}, Texto: {texto}"
        )


####  CONVERTIR TXT A JSON

import json
import re


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


# Ruta al archivo de texto
file_path = "./documents/mf3.txt"

# Convertir el archivo de texto a JSON
json_data = convert_text_to_json(file_path)

# Guardar el resultado en un archivo JSON
output_file_path = "./documents/mf3_output.json"
with open(output_file_path, "w", encoding="utf-8") as output_file:
    json.dump(json_data, output_file, ensure_ascii=False, indent=4)

print(f"Archivo JSON generado en: {output_file_path}")


def get_ids_and_question(json_data):
    pregs = [[id, json_data.get(id).get("Pregunta")] for id in json_data.keys()]
    return pregs
