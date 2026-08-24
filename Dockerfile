FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY docs ./docs
COPY examples ./examples
COPY config.example.json ./config.example.json

RUN pip install --no-cache-dir .

RUN mkdir -p /app/data

EXPOSE 8080

CMD ["loopgraph-supervisor", "serve", "--config", "config.example.json", "--host", "0.0.0.0"]

