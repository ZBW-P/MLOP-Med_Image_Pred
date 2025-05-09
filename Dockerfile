FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt mlflow

COPY model.pth .
COPY app.py .
COPY model.py .
COPY static/ ./static/
EXPOSE 8265
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8265"]
