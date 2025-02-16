import os
import re

import ollama
import streamlit as st
from PyPDF2 import PdfReader

# List of available models
models = {
    "llama3.1:8b": "42182419e950",
    "mistral:7b": "f974a74358d6",
    "qwen2.5-coder:7b": "2b0496514337",
}


def clean_response(text: str) -> str:
    """Clean a text

    Args:
        text (str): Some text inside a tag

    Returns:
        str: Cleaned text
    """
    return text.message.content


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
        if "Versat" in message:
            print("PDF content was considered")
            prompt = f"""
                Contexto de la conversación:
                {conversation_history}
                \n\n
                Texto del usuario: {pdf_content}
                \n\n
                Pregunta del usuario: {message}
            """
        else:
            prompt = f"""
                Contexto de la conversación:
                {conversation_history}
                
                \n\n
                Pregunta del usuario: {message}
            """
        response = ollama.chat(
            model=st.session_state.selected_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response
    except Exception as e:
        return f"Error {str(e)}"


def extract_text_from_pdfs(folder_path: str) -> str:
    """Extract text from all PDFs in a folder

    Args:
        folder_path (str): Path to the folder containing PDFs

    Returns:
        str: Concatenated text from all PDFs
    """
    pdf_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            pdf_reader = PdfReader(file_path)
            for page in pdf_reader.pages:
                pdf_text += page.extract_text()
    return pdf_text


def main():
    """
    Write the application
    """
    st.title("Habla con Versat")

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

    # Extract text from PDFs in the 'documents' folder
    pdf_content = extract_text_from_pdfs("documents")

    # Show chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if not st.session_state.processing:
        prompt = st.chat_input(
            "Escribe tu mensaje aquí...", disabled=st.session_state.processing
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
                assistant_response = clean_response(response)
                st.write(assistant_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_response}
                )
                # Reset processing state
                st.session_state.processing = False
                st.rerun()  # Rerun to update UI


if __name__ == "__main__":
    main()
