# Hermes — Weather Trading Agent

Automated agent that generates trading signals from weather data and trades weather-related markets on Polymarket.

## Setup

```bash
# Install dependencies
pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env
```

## Quick Start

```bash
# Run the demo pipeline: fetch weather → scan markets
python demo.py

# Run tests
pytest -v
```

## Project Structure

| Module | Responsibility |
|---|---|
| `polymarket/` | Gamma API client, weather market discovery & scoring |
| `weather_mcp/` | Weather data fetching via wttr.in |
| `agent/` | Decision engine (planned) |
| `common/` | Configuration, logging |

## Status

- [x] Project skeleton
- [x] Polymarket market discovery (read-only)
- [x] Weather data fetching
- [ ] Decision engine
- [ ] Trade execution
