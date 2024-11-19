FROM python:3.12-slim

EXPOSE 8501

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD streamlit run data_app.py --server.port=8502 --browser.serverAddress="0.0.0.0"
