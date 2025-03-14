import streamlit as st
from model_processing import (
    load_document,
    parse_document,
    setup_qa_chain,
)

# Configuración inicial de Streamlit
st.set_page_config(
    page_title="Chatbot de Soporte Técnico",
    page_icon="🤖",
    layout="wide",
)

# Título de la aplicación
st.title("🤖 Chatbot de Soporte Técnico")
st.markdown("Bienvenido al chatbot de soporte técnico. Haz preguntas relacionadas con el sistema Versat Sarasola.")

# Ruta del archivo
file_path = "./documents/mf.txt"

# Leer el documento
raw_text = load_document(file_path)
if not raw_text:
    st.error("❌ No se pudo cargar el archivo de documentación.")
else:
    # Procesar el documento
    chunks = parse_document(raw_text)
    if not chunks:
        st.error("❌ No se encontraron preguntas válidas en el documento.")
    else:
        # Mostrar información sobre el procesamiento
        st.success("✅ Documento procesado exitosamente.")
        st.info(f"Se extrajeron {len(chunks)} bloques de información del documento.")

        # Obtener la lista de modelos disponibles en Ollama
        try:
            from langchain_ollama import OllamaLLM

            # Lista de modelos disponibles (puedes personalizarla según tus necesidades)
            available_models = ["qwen2.5:3B", "llama3", "mistral", "phi"]
            selected_model = st.selectbox("Selecciona un modelo:", available_models)

            # Configurar la cadena QA con el modelo seleccionado
            qa_chain = setup_qa_chain(selected_model)

            # Iniciar ciclo interactivo de consultas
            st.subheader("Haz tu pregunta:")
            user_query = st.text_input("Escribe tu pregunta aquí:")

            if user_query:
                if user_query.lower() in ["salir", "exit"]:
                    st.info("👋 ¡Hasta luego!")
                else:
                    # Obtener respuesta del modelo
                    with st.spinner(f"⏳ Procesando tu pregunta con el modelo '{selected_model}'..."):
                        response = qa_chain.invoke(user_query)
                    st.subheader("✅ Respuesta:")
                    st.write(response.strip())
        except Exception as e:
            st.error(f"❌ Error al configurar el modelo: {e}")
