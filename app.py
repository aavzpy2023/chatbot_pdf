import ollama
import requests
import streamlit as st

from processing import (
    get_answer_from_model,
    get_milvus_client,
    get_question_contents,
    print_with_date,
)

# Initialize session state for history and last processed question
if "history" not in st.session_state:
    st.session_state.history = []
if "last_processed_question" not in st.session_state:
    st.session_state.last_processed_question = None


def configure_sidebar():
    """
    Configures the sidebar with a dropdown to select the model.

    Returns:
        str: The selected model from the dropdown.
    """
    with st.sidebar:
        st.header("Configuración")
        try:
            # Dynamically fetch available models from Ollama
            available_models = [model["model"] for model in ollama.list()["models"]]
            selected_model = st.selectbox(
                "Selecciona un modelo:", available_models, key="model_selection"
            )

            return selected_model
        except Exception as e:
            print_with_date(f"❌ Error to load models: {e}")
            return None


def get_embedding_ollama(text: str):
    """
    Sends a POST request to the Ollama API to convert text into an embedding.

    text: list of texts: ["your text"]
    Returns:
        list: The embedding vector as a list of floats.
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
    """
    Check if the provided string is a valid JSON.

    Args:
        json_data (str): The string to be checked.

    Returns:
        bool: True if the string is a valid JSON, False otherwise.
    """
    try:
        # Try to load the JSON data
        json.loads(json_data)
        return True
    except ValueError as e:
        # Catch errors of JSON format
        print_with_date(f"The JSON is invalid: {e}")
        return False


def main():
    """
    Main function to run the Versat Virtual Assistant application.
    """
    # Configure Streamlit page
    st.set_page_config(
        page_title="Asistente Virtual Versat",
        page_icon=" 📚",
        layout="wide",
    )

    # Title and description
    st.title("📚 Asistente Virtual Versat")
    st.markdown("Bienvenido al Asistente Virtual Versat.")

    # Sidebar configuration
    selected_model = configure_sidebar()
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = selected_model

    if not selected_model:
        st.stop()

    # Interactive query handling
    user_query = st.chat_input("Escribe tu pregunta aquí")

    if user_query:
        if user_query.lower() in ["salir", "exit"]:
            st.info("👋 ¡Hasta luego!")
        else:
            with st.spinner(
                "Procesando su pregunta..."
            ):  # Get the embedding of the question
                vt_search = get_embedding_ollama(user_query)
                if not vt_search or not isinstance(vt_search, list):
                    st.error(
                        "El proceso de embeddings ha fallado o devolvió datos inválidos."
                    )
                    st.stop()

                # Search in Milvus
                try:
                    print_with_date("Searching in Milvus...")
                    # Connect to Milvus
                    client = get_milvus_client(
                        db_name="versat", collection_name="sarasola"
                    )
                    res_query = client.search(
                        collection_name="sarasola",
                        anns_field="q_vector",
                        data=[vt_search],
                        limit=10,
                        search_params={"metric_type": "COSINE"},
                    )[0]
                    id_interest = [item.get("id") for item in res_query]
                    print_with_date(f"Selected IDS: {id_interest}")
                    if not id_interest:
                        print_with_date("The IDs were not found in the file.")
                        st.stop()
                    id_interest = list(set(id_interest))
                    resultados = "\n\n".join(get_question_contents(id_interest))
                except Exception as e:
                    print_with_date(f"Error during the search: {e}")
                    st.stop()

                # Build the prompt
                prompt = f"""
                **Rol:** Eres un asistente experto que responde preguntas sobre el software basándose *únicamente* en fragmentos de documentación proporcionados.

                **Tarea:** Analiza el siguiente "Contexto", que puede contener uno o más fragmentos relevantes. Responde la "Pregunta del Usuario" de forma precisa y útil.

                **Instrucciones:**
                1.  Para cada fragmento en el contexto, localiza la información más relevante para la pregunta, priorizando `Sct. Respuesta` y `Sct. Pasos a Seguir`.
                2.  **Sintetiza** la información de los fragmentos relevantes en una **única respuesta coherente**. No te limites a listar las respuestas de cada fragmento por separado.
                3.  La respuesta debe ser clara, directa y enfocada en resolver la duda del usuario.
                4.  Basa tu respuesta *exclusivamente* en el contexto. No inventes información ni uses conocimiento externo.
                5.  Si el contexto no contiene la información necesaria, indícalo claramente.

                **Contexto:**

                Contexto:
                {resultados}

                Pregunta: {user_query}

                Respuesta:
                """

            print_with_date(f"Building the final answer by {selected_model}...")

            # Get the answer from the model
            output = get_answer_from_model(model=selected_model, prompt=prompt)
            print_with_date(f"The answer has been generated")

            # Save question and answer to history
            st.session_state.history.append({"question": user_query, "answer": output})

            # Update the last processed question
            st.session_state.last_processed_question = user_query

            # Display chat history
            if st.session_state.history:
                st.subheader("Historial de Preguntas y Respuestas:")
                for entry in st.session_state.history:
                    st.markdown(f"**Pregunta:** {entry['question']}")
                    st.markdown(f"**Respuesta:** {entry['answer']}")
                    st.markdown("---")
            else:
                st.session_state.historial = []


if __name__ == "__main__":
    main()
