import streamlit as st
from model_processing import load_document, parse_document, setup_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import json

# Initialize session state for history and last processed question
if "history" not in st.session_state:
    st.session_state.history = []
if "last_processed_question" not in st.session_state:
    st.session_state.last_processed_question = None

@st.cache_data
def load_and_process_document(file_path):
    """
    Loads and processes the document into chunks for further use.

    Args:
        file_path (str): Path to the document file.

    Returns:
        list: List of document chunks if successful, None otherwise.
    """
    raw_text = load_document(file_path)
    if not raw_text:
        print("❌ No se pudo cargar el archivo de documentación.")
        return None

    chunks = parse_document(raw_text)
    if not chunks:
        print("❌ No se encontraron preguntas válidas en el documento.")
        return None

    print("✅ Documento procesado exitosamente.")
    print(f"Se extrajeron {len(chunks)} bloques de información del documento.")
    return chunks

@st.cache_resource
def setup_vector_store(chunks, selected_model):
    """
    Sets up the vector store using embeddings from the selected model.

    Args:
        chunks (list): List of document chunks.
        selected_model (str): Selected model name.

    Returns:
        FAISS: Vector store object if successful, None otherwise.
    """
    try:
        embeddings = OllamaEmbeddings(model=selected_model)
        vector_store = FAISS.from_texts(chunks, embeddings)
        print("✅ Vector Store configurado exitosamente.")
        return vector_store
    except Exception as e:
        print(f"❌ Error al configurar el vector store: {e}")
        return None

@st.cache_resource
def initialize_qa_chain(_vector_store, selected_model):
    """
    Initializes the QA chain with the vector store and selected model.

    Args:
        _vector_store (FAISS): Vector store object. The underscore tells Streamlit not to hash this argument.
        selected_model (str): Selected model name.

    Returns:
        RetrievalQA: QA chain object if successful, None otherwise.
    """
    try:
        llm = OllamaLLM(model=selected_model)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=_vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template="""
                    Eres un asistente experto en el sistema Versat Sarasola.
                    Responde únicamente usando la información proporcionada en el contexto.
                    Si no hay información relevante, responde: "⚠️ No tengo información sobre este tema."
                    Si la respuesta es corta, responda con una respuesta corta
                    Responde de manera precisa


                    Contexto:
                    {context}

                    Pregunta:
                    {question}

                    Respuesta:
                    """,
                    input_variables=["context", "question"]
                )
            }
        )
        print("✅ QA Chain inicializada exitosamente.")
        return qa_chain
    except Exception as e:
        print(f"❌ Error al inicializar la cadena QA: {e}")
        return None

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
            print(f"❌ Error al cargar los modelos: {e}")
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
            formatted_response = response.get("result", "No se encontró una respuesta clara.")
        elif isinstance(response, str):
            try:
                parsed_response = json.loads(response)
                formatted_response = parsed_response.get("result", "No se encontró una respuesta clara.")
            except (json.JSONDecodeError, AttributeError):
                formatted_response = response
        else:
            formatted_response = str(response)

        return formatted_response
    except Exception as e:
        print(f"❌ Error al procesar la pregunta: {e}")
        return "⚠️ Ocurrió un error al procesar tu pregunta."

def main():
    """
    Main function to run the Versat Virtual Assistant application.
    """
    # Configure Streamlit page
    st.set_page_config(
        page_title="Asistente Virtual Versat",
        page_icon="🤖",
        layout="wide",
    )

    # Title and description
    st.title("🤖 Asistente Virtual Versat")
    st.markdown("Bienvenido al Asistente Virtual Versat. Haz preguntas relacionadas con el sistema Versat Sarasola.")

    # Sidebar configuration
    selected_model = configure_sidebar()
    if not selected_model:
        st.stop()

    # Load and process document
    file_path = "./documents/mf.txt"
    chunks = load_and_process_document(file_path)
    if not chunks:
        st.stop()

    # Set up vector store and QA chain
    vector_store = setup_vector_store(chunks, selected_model)
    if not vector_store:
        st.stop()

    qa_chain = initialize_qa_chain(vector_store, selected_model)
    if not qa_chain:
        st.stop()

    # Display chat history
    st.subheader("Historial de Preguntas y Respuestas:")
    for entry in st.session_state.history:
        st.markdown(f"**Pregunta:** {entry['question']}")
        st.markdown(f"**Respuesta:** {entry['answer']}")
        st.markdown("---")

    # Interactive query handling
    user_query = st.text_input("Pregunta:", placeholder="Escribe tu pregunta aquí")

    if user_query:
        if user_query.lower() in ["salir", "exit"]:
            st.info("👋 ¡Hasta luego!")
        else:
            # Check if the question has already been processed
            if user_query != st.session_state.last_processed_question:
                output = process_user_query(qa_chain, user_query)

                # Save question and answer to history
                st.session_state.history.append({"question": user_query, "answer": output})

                # Update the last processed question
                st.session_state.last_processed_question = user_query

                # Clear the textbox after processing the query
                st.query_params
                st.rerun()

if __name__ == "__main__":
    main()
