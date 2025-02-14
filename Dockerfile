FROM python:3.12-alpine

# Instalar dependencias del sistema
RUN apk add --no-cache \
    build-base \
    libffi-dev \
    openssl-dev \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt 

WORKDIR /app

COPY . .

CMD ["streamlit", "run", "data_app.py", "--server.port=8502", "--browser.serverAddress=0.0.0.0"]
