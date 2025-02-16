import re

import ollama
import streamlit as st


def clean_response(text: str) -> str:
    """Clean a text

    Args:
        text (str): Some text inside a tag

    Returns:
        str: Cleaned text
    """
    return text.message.content


def get_api_response(message: str) -> str:
    """Get response from API

    Args:
        message (str): message to API

    Returns:
        str: The response of the API
    """
    try:
        response = ollama.chat(
            model="llama3.1:8b", messages=[{"role": "user", "content": message}]
        )
        return response
    except Exception as e:
        return f"Error {str(e)}"


def main():
    """
    Write the application
    """
    all_message = []
    st.title("Habla con Versat")
    with st.sidebar:
        if st.button("Limpiar chat"):
            st.session_state.messages = []
            # st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # USer input
    if prompt := st.chat_input("Escribe tu mensaje aqui..."):

        # Add user message to the chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from AI
        with st.chat_message("assistant"):
            response = get_api_response(prompt)
            print(response)
            st.write(clean_response(response))
            st.session_state.messages.append(
                {"role": "assistant", "content": clean_response(response)}
            )
            all_message.append(clean_response(response))


if __name__ == "__main__":
    main()
