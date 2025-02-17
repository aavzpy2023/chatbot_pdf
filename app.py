import os
import time

import ollama
import streamlit as st
from docx import Document
from PyPDF2 import PdfReader

# List of available models
models = {
    "mistral:7b": "f974a74358d6",
    "llama3.1:8b": "42182419e950",
    "qwen2.5-coder:7b": "2b0496514337",
    "llama3.1:latest": "42182419e950",
}


def clean_response(text: str) -> str:
    """Clean a text

    Args:
        text (str): Some text inside a tag

    Returns:
        str: Cleaned text
    """
    return text.message.content.replace("VersatInstalacion.exe", "Versat")


def get_api_response(message: str, pdf_content: str = "") -> str:
    """Get response from API

    Args:
        message (str): message to API
        pdf_content (str): content extracted from PDFs

    Returns:
        str: The response of the API
    """
    try:
        # Include conversation history in the prompt
        conversation_history = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages]
        )
        # with open("./prompt.txt", "r") as f:
        #     prompt = f.read()
        #     prompt = prompt.replace("[conversation_history]", conversation_history)
        #     prompt = prompt.replace("[pdf_content]", pdf_content)
        #     prompt = prompt.replace("[message]", message)
        prompt = f"""
            Proporciona respuestas directas y seguras basadas en el contexto proporcionado de forma cortes y profesional. Evita usar términos como "parece" o "según la información proporcionada". Responde de manera clara y concisa.

            Contexto de la conversación:
            {conversation_history}

            Texto del usuario: {pdf_content}
            \n\n
            Pregunta del usuario: {message.replace('Sarasola', 'Versat')}
        # """

        response = ollama.chat(
            model=st.session_state.selected_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response
    except Exception as e:
        return f"Error {str(e)}"


def extract_text_from_file(file_path: str) -> str:
    """Extract text from a file (PDF, DOCX, or TXT)

    Args:
        file_path (str): Path to the file

    Returns:
        str: Extracted text from the file
    """
    if file_path.endswith(".pdf"):
        pdf_reader = PdfReader(file_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        return text
    return ""


def extract_text_from_folder(folder_path: str) -> str:
    """Extract text from all files in a folder

    Args:
        folder_path (str): Path to the folder containing files

    Returns:
        str: Concatenated text from all files
    """
    combined_text = ""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith((".pdf", ".docx", ".txt")):
            combined_text += extract_text_from_file(file_path) + "\n"
    return combined_text


def save_context_to_file(file_path: str, content: str):
    """Save the extracted content to a file

    Args:
        file_path (str): Path to the file to save the content
        content (str): Content to be saved
    """
    print("guardando contexto", file_path)
    content = content.replace("Versat Sarasola", "Versat")
    content = content.replace("Versat", "Versat Sarasola")
    content = content.replace(
        "Versat Sarasola", "Versat Sarasola tambien conocido como Sarasola o  Versat"
    )
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)


def get_folder_modification_time(folder_path: str) -> float:
    """Get the latest modification time of files in a folder

    Args:
        folder_path (str): Path to the folder

    Returns:
        float: The latest modification time
    """
    modification_times = [
        os.path.getmtime(os.path.join(folder_path, f)) for f in os.listdir(folder_path)
    ]
    return max(modification_times) if modification_times else 0


def check_and_update_context(folder_path: str, context_file_path: str):
    """Check for changes in the folder and update the context file if modified

    Args:
        folder_path (str): Path to the folder to monitor
        context_file_path (str): Path to the context file to update
    """
    # Check if the context file exists
    if not os.path.exists(context_file_path):
        # Initial extraction and saving
        combined_text = extract_text_from_folder(folder_path)
        save_context_to_file(context_file_path, combined_text)

    # Get the latest modification time
    last_modification_time = get_folder_modification_time(folder_path)
    context_modification_time = os.path.getmtime(context_file_path)

    if last_modification_time > context_modification_time:
        # Update the context file if the folder has been modified
        combined_text = extract_text_from_folder(folder_path)
        save_context_to_file(context_file_path, combined_text)


def main():
    """
    Write the application
    """
    st.title("Asistente Versat")

    # Initialize session state for conversation history and processing state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # Sidebar for model selection and clearing chat
    with st.sidebar:
        # Model selection
        selected_model = st.selectbox(
            "Selecciona un modelo:",
            options=list(models.keys()),
            disabled=st.session_state.processing,
        )
        st.session_state.selected_model = selected_model

        # Center the button and add rounded corners using custom CSS
        st.markdown(
            """
            <style>
            .center-button {
                display: flex;
                justify-content: center;
                margin-top: 20px;
            }
            .stButton>button {
                border-radius: 15px;
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                border: none;
                cursor: pointer;
                font-size: 16px;
            }
            .stButton>button:hover {
                background-color: #45a049;
            }
            </style>
            <div class="center-button">
            """,
            unsafe_allow_html=True,
        )

        # Clear chat button
        if st.sidebar.button("Limpiar chat"):
            # Clear the chat history in the session state
            st.session_state.messages = []

        st.markdown("</div>", unsafe_allow_html=True)

    # Paths
    folder_path = "documents"
    context_file_path = "context.txt"

    # Check and update context
    print("verificando archivos")
    check_and_update_context(folder_path, context_file_path)

    # Read context from file
    with open(context_file_path, "r", encoding="utf-8") as file:
        pdf_content = file.read()

    # Show chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if not st.session_state.processing:
        prompt = st.chat_input(
            "Escribe su pregunta o consulta aquí...",
            disabled=st.session_state.processing,
        )
        if prompt:
            # Set processing state to True
            st.session_state.processing = True
            # Add user message to the chat
            st.session_state.messages.append(
                {"role": "user", "content": prompt.capitalize()}
            )
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get response from AI
            with st.chat_message("assistant"):
                response = get_api_response(prompt, pdf_content)
                print(response)
                assistant_response = clean_response(response).replace(
                    "Respuesta del modelo:", ""
                )
                st.write(assistant_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_response}
                )
                # Reset processing state
                st.session_state.processing = False
                st.rerun()  # Rerun to update UI


if __name__ == "__main__":
    main()
