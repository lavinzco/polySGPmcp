FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY pyproject.toml .
COPY agent/ agent/
COPY common/ common/
COPY polymarket/ polymarket/
COPY weather_mcp/ weather_mcp/
COPY scheduler/ scheduler/
COPY scripts/ scripts/

RUN pip install --no-cache-dir -e .

# Build-time import check: fail fast if any dependency is missing
RUN python -c "\
import httpx; \
import pydantic; \
import pydantic_settings; \
import openai; \
import anthropic; \
import dotenv; \
from agent.calibration.settlement_tracker import check_settled_markets_singapore; \
from scheduler.cron import run_collection_once; \
print('All imports OK')"

RUN mkdir -p /app/data

ENV DATA_DIR=/app/data
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

CMD ["python", "-m", "scheduler.cron"]
