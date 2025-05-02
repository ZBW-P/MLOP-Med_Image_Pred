FROM python:3.11-slim

RUN pip install --no-cache-dir requests pyyaml scikit-learn

WORKDIR /app
COPY datamerged2.py /app/
COPY datasets_config.yaml /app/

CMD ["python", "datamerged2.py"]
