from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import re

# Plantilla de prompt personalizada
PROMPT_TEMPLATE = """
Eres un asistente experto en el sistema. Responde únicamente usando la información proporcionada en el contexto.
Si no hay información relevante, responde: "⚠️ No tengo información sobre este tema."
Debes responder siempre en espanol y solo mostrar la respuesta final en la interface


Contexto:
{context}

Pregunta:
{question}
"""

# =============================================================================
# Función para leer el documento
# =============================================================================
def load_document(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"❌ Error: El archivo '{file_path}' no se encontró.")
        return None
    except Exception as e:
        print(f"❌ Error al leer el archivo: {e}")
        return None

# =============================================================================
# Función para extraer detalles de un bloque de texto
# =============================================================================
def extract_details(content):
    id_match = re.search(r'ID:\s*(.+)', content)
    titulo_match = re.search(r'Título:\s*(.+)', content)
    categoria_match = re.search(r'Categoría:\s*(.+)', content)
    prioridad_match = re.search(r'Prioridad:\s*(.+)', content)
    version_match = re.search(r'Versión:\s*(.+)', content)
    fecha_match = re.search(r'Fecha de Actualización:\s*(.+)', content)
    funcionalidad_match = re.search(r'Funcionalidad:\s*([\s\S]+?)(?=Respuesta:|Roles de Acceso:|$)', content)
    respuesta_match = re.search(r'Respuesta:\s*([\s\S]+?)(?=Roles de Acceso:|Variantes de Preguntas:|Pasos a Seguir:|Notas Adicionales:|Errores Comunes y Soluciones:|Prerrequisitos:|Resultados Esperados:|Referencias Relacionadas:|Escenarios Posibles:|Casos Especiales:|$)', content)
    roles_match = re.search(r'Roles de Acceso:\s*([\s\S]+?)(?=Variantes de Preguntas:|Pasos a Seguir:|Notas Adicionales:|Errores Comunes y Soluciones:|Prerrequisitos:|Resultados Esperados:|Referencias Relacionadas:|Escenarios Posibles:|Casos Especiales:|$)', content)
    variantes_match = re.findall(r'¿(.*?)\?(.*?)(?=¿|\Z)', content, re.DOTALL)
    pasos_match = re.search(r'Pasos a Seguir:\s*([\s\S]+?)(?=Notas Adicionales:|Errores Comunes y Soluciones:|Prerrequisitos:|Resultados Esperados:|Referencias Relacionadas:|Escenarios Posibles:|Casos Especiales:|$)', content)
    errores_match = re.findall(r'"(.*?)":\s*(.*?)\n', content, re.DOTALL)
    resultados_match = re.search(r'Resultados Esperados:\s*([\s\S]+?)(?=Referencias Relacionadas:|Escenarios Posibles:|Casos Especiales:|$)', content)
    escenarios_match = re.findall(r'Escenarios Posibles:\s*([\s\S]+?)(?=Casos Especiales:|Feedback del Usuario:|$)', content, re.DOTALL)
    casos_especiales_match = re.search(r'Casos Especiales:\s*([\s\S]+?)(?=Feedback del Usuario:|$)', content)
    feedback_match = re.search(r'Feedback del Usuario:\s*([\s\S]+)', content)

    return {
        "id": id_match.group(1).strip() if id_match else "No especificado.",
        "titulo": titulo_match.group(1).strip() if titulo_match else "No especificado.",
        "categoria": categoria_match.group(1).strip() if categoria_match else "No especificada.",
        "prioridad": prioridad_match.group(1).strip() if prioridad_match else "No especificada.",
        "version": version_match.group(1).strip() if version_match else "No especificada.",
        "fecha": fecha_match.group(1).strip() if fecha_match else "No especificada.",
        "funcionalidad": funcionalidad_match.group(1).strip() if funcionalidad_match else "No especificada.",
        "respuesta": respuesta_match.group(1).strip() if respuesta_match else "No especificada.",
        "roles": roles_match.group(1).strip() if roles_match else "No especificados.",
        "variantes": [{"pregunta": v[0].strip(), "respuesta": v[1].strip()} for v in variantes_match] if variantes_match else [],
        "pasos": [p.strip() for p in pasos_match.group(1).splitlines()] if pasos_match else [],
        "errores": [(e[0].strip(), e[1].strip()) for e in errores_match] if errores_match else [],
        "resultados": resultados_match.group(1).strip() if resultados_match else "No especificados.",
        "escenarios": [e.strip() for e in escenarios_match] if escenarios_match else [],
        "casos_especiales": casos_especiales_match.group(1).strip() if casos_especiales_match else "No especificados.",
        "feedback": feedback_match.group(1).strip() if feedback_match else "No especificado."
    }

# =============================================================================
# Función para dividir el documento en chunks basados en preguntas
# =============================================================================
def parse_document(text):
    chunks = []
    try:
        # Dividir el texto en bloques basados en "ID:"
        preguntas = re.split(r'(ID:\s*\S+)', text, flags=re.IGNORECASE)

        for i in range(1, len(preguntas), 2):
            id_titulo = preguntas[i].strip()  # "ID: ..."
            contenido = preguntas[i + 1].strip()  # Contenido asociado

            # Extraer detalles del contenido
            detalles = extract_details(id_titulo + "\n" + contenido)

            # Construir el chunk final
            chunk = f"""
            ID: {detalles["id"]}
            Título: {detalles["titulo"]}
            Categoría: {detalles["categoria"]}
            Prioridad: {detalles["prioridad"]}
            Versión: {detalles["version"]}
            Fecha de Actualización: {detalles["fecha"]}

            Funcionalidad:
            {detalles["funcionalidad"]}

            Respuesta:
            {detalles["respuesta"]}

            Roles de Acceso:
            {detalles["roles"]}

            Variantes de Preguntas:
            {"; ".join([f"{v['pregunta']} -> {v['respuesta']}" for v in detalles["variantes"]])}

            Pasos a Seguir:
            {"; ".join(detalles["pasos"])}

            Errores Comunes y Soluciones:
            {"; ".join([f"{e[0]}: {e[1]}" for e in detalles["errores"]])}

            Resultados Esperados:
            {detalles["resultados"]}

            Escenarios Posibles:
            {", ".join(detalles["escenarios"])}

            Casos Especiales:
            {detalles["casos_especiales"]}

            Feedback del Usuario:
            {detalles["feedback"]}
            """.strip()

            chunks.append(chunk)

    except Exception as e:
        print(f"❌ Error procesando el documento: {e}")

    return chunks

def setup_vector_store(chunks, model_name):
    embeddings = OllamaEmbeddings(model=model_name)
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store



# =============================================================================
# Función para configurar la cadena QA con el modelo seleccionado
# =============================================================================
def setup_qa_chain(vector_store, model_name):
    llm = OllamaLLM(model=model_name)
    prompt_template = """
    Eres un asistente experto en el sistema Versat Sarasola.
    Contexto:
    {context}

    Pregunta:
    {question}

    Respuesta:
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
        }
    )
    return qa_chain
