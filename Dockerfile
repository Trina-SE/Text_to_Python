FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt pyproject.toml /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY src /app/src
COPY run_all.py /app/run_all.py

ENV PYTHONPATH=/app/src

ENTRYPOINT ["python", "/app/run_all.py"]
