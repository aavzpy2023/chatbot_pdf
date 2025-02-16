FROM python:3.12-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar solo el archivo requirements.txt primero
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install -r requirements.txt

# Copiar el resto de los archivos
COPY . .

CMD ["streamlit", "run", "data_app.py", "--server.port=8502", "--browser.serverAddress=0.0.0.0"]
