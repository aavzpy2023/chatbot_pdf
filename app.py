import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from model_processing import crear_vector_store, create_model, MODELOS, PROMPT_TEMPLATE


def main():
    st.set_page_config(page_title="Asistente Versat Sarasola", layout="wide")
    st.title("📚 Asistente Versat Sarasola")

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    # Sidebar con modelos disponibles
    with st.sidebar:
        modelo_seleccionado = st.selectbox(
            "Modelo:",
            options=list(MODELOS.keys()),
            format_func=lambda x: MODELOS[x],
            index=0  # Modelo por defecto: Qwen2.5 Coder 7B
        )

    # Configuración del LLM con Ollama
    llm = create_model(modelo_seleccionado)

    # Crear QA chain
    vector_store = crear_vector_store()
    if vector_store:
        PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
        )

    # Interfaz de chat
    if "historial" not in st.session_state:
        st.session_state.historial = []

    for mensaje in st.session_state.historial:
        with st.chat_message(mensaje["role"]):
            st.write(mensaje["content"])

    pregunta = st.chat_input("Escribe tu pregunta sobre Versat Sarasola...")

    if pregunta and st.session_state.qa_chain:
        with st.chat_message("user"):
            st.write(pregunta)

        try:
            with st.spinner("Analizando..."):
                respuesta = st.session_state.qa_chain.invoke(pregunta)
                respuesta_limpia = respuesta.get("result", "No disponible")

            with st.chat_message("assistant"):
                st.write(respuesta_limpia)

            st.session_state.historial.extend([
                {"role": "user", "content": pregunta},
                {"role": "assistant", "content": respuesta_limpia},
            ])

        except Exception as e:
            st.error("Error: Reinicia el servidor de Ollama" + str(e))
            st.session_state.historial.append({
                "role": "assistant",
                "content": "Ocurrió un error. Verifica que Ollama esté activo."
            })

if __name__ == "__main__":
    main()
