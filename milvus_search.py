from pymilvus import MilvusClient

from milvus_insert_data import get_embeddings


def connect_to_milvus_db(db_name: str):
    client = MilvusClient(uri="http://localhost:19530")
    client.using_database(db_name=db_name)  # Activar la base de datos
    return client


def search_vector(client, collection_name: str, vector: list):
    res = client.search(
        collection_name=collection_name,
        anns_field="q_vector",
        data=[vector],
        limit=2,
        search_params={"metric_type": "L2"},
    )
    return res


# def check_index_metric(client, collection_name):
#     collection = client.describe_collection(collection_name)
#     for index in collection["indexes"]:
#         print(
#             f"Índice: {index['field_name']}, Metric Type: {index['params']['metric_type']}"
#         )


# if __name__ == "__main__":
#     mv_client = connect_to_milvus_db("versat")
#     check_index_metric(mv_client, "sarasola")


if __name__ == "__main__":
    mv_client = connect_to_milvus_db("versat")
    question = input("Type your question: ")
    question_embed = get_embeddings([question])[0][0]

    search_res = search_vector(mv_client, "sarasola", question_embed)
    print(search_res)
