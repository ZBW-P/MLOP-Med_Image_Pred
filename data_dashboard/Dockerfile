# FROM python:3.9-slim

# WORKDIR /app

# COPY . .

# RUN pip install --no-cache-dir -r requirements.txt

# EXPOSE 8501

# CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir \
    streamlit pandas pillow python-swiftclient

EXPOSE 9002

CMD ["streamlit", "run", "app2.py", "--server.address=0.0.0.0", "--server.port=9002"]


