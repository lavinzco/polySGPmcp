FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY pyproject.toml .
COPY agent/ agent/
COPY common/ common/
COPY polymarket/ polymarket/
COPY weather_mcp/ weather_mcp/
COPY scheduler/ scheduler/

RUN pip install --no-cache-dir -e .

RUN mkdir -p /app/data

ENV DATA_DIR=/app/data
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

CMD ["python", "-m", "scheduler.cron"]
