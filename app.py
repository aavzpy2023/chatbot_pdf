import streamlit as st
from model_processing import load_and_process_document, PROMPT_TEMPLATE, print_with_date, create_model, initialize_qa_chain, setup_vector_store
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import json
import os
import datetime

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
            import ollama

            # Dynamically fetch available models from Ollama
            available_models = [model['model'] for model in ollama.list()['models']]
            selected_model = st.selectbox(
                "Selecciona un modelo:",
                available_models,
                key="model_selection"
            )
            return selected_model
        except Exception as e:
            print_with_date(f"❌ Error al cargar los modelos: {e}")
            return None

def process_user_query(qa_chain, user_query):
    """
    Processes the user's query using the QA chain and returns the response.

    Args:
        qa_chain (RetrievalQA): QA chain object.
        user_query (str): User's question.

    Returns:
        str: Formatted response to the user's query.
    """
    try:
        response = qa_chain.invoke(user_query)

        # Handle JSON or plain text responses
        if isinstance(response, dict):
            print_with_date("Response as dict")
            formatted_response = response.get("result", "No se encontró una respuesta clara.")
        elif isinstance(response, str):
            print_with_date("Response as str")
            try:
                parsed_response = json.loads(response)
                formatted_response = parsed_response.get("result", "No se encontró una respuesta clara.")
            except (json.JSONDecodeError, AttributeError):
                formatted_response = response
        else:
            formatted_response = str(response)

        return formatted_response
    except Exception as e:
        print_with_date(f"❌ Error al procesar la pregunta: {e}")
        return "⚠️ Ocurrió un error al procesar tu pregunta."

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

    # Set up vector store and QA chain
    if "qa_chain" not in st.session_state or st.session_state.selected_model != selected_model:
        # Load and process document
        file_path = os.getenv("CONTEXT_FILE", "./documents/.txt")

        print_with_date(f'processing informations of {file_path}')
        chunks = load_and_process_document(file_path)
        if not chunks:
            st.stop()

        print_with_date('creating vector store')
        vector_store = setup_vector_store(chunks, selected_model)
        if not vector_store:
            st.stop()

    if "qa_chain" not in st.session_state:
        print_with_date('initilizing qa chain')
        st.session_state.qa_chain = initialize_qa_chain(vector_store, selected_model)

    if not st.session_state.qa_chain:
        st.stop()


    # Interactive query handling
    user_query = st.chat_input("Escribe tu pregunta aquí")

    if user_query:
        if user_query.lower() in ["salir", "exit"]:
            st.info("👋 ¡Hasta luego!")
        else:
            # Check if the question has already been processed
            if user_query != st.session_state.last_processed_question:
                qa_chain = st.session_state.qa_chain
                with st.spinner("Procesando su pregunta"):
                    print_with_date(f"Procesando pregunta con el modelo {selected_model}")
                    output = process_user_query(qa_chain, user_query)

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

                # Clear the textbox after processing the query
                # st.query_params
                # st.rerun()

if __name__ == "__main__":
    main()
