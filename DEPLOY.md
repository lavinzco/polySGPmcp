# Hermes VPS Deployment

## Quick Start

```bash
# 1. Get the code onto your VPS
git clone <your-repo-url> hermes
cd hermes

# 2. Configure environment
cp .env.example .env
nano .env  # fill in API keys (see Required Keys below)

# 3. Create data directory
mkdir -p data

# 4. Start
docker-compose up -d
```

## Required Keys

The `.env` file needs at minimum:

| Variable | Description |
|---|---|
| `STRATEGY_API_KEY` | DeepSeek or other LLM provider API key |
| `CLASSIFICATION_API_KEY` | Same or different LLM key for market classification |

All other settings have sensible defaults. See `.env.example` for the full list.

## Operations

```bash
# View logs (follow mode)
docker-compose logs -f

# Check health — last successful run time
cat data/last_run.txt

# Docker health status
docker inspect --format='{{.State.Health.Status}}' hermes-scheduler

# Stop
docker-compose down

# Rebuild after code changes
docker-compose up -d --build

# Run one-off settlement check
docker-compose exec hermes python -m scheduler.cron --settle

# Run one collection cycle manually
docker-compose exec hermes python -m scheduler.cron --once --mode singapore
```

## Data & Backups

All persistent data lives in `./data/`:

| File | Contents |
|---|---|
| `hermes_decisions.db` | Decision log (evaluations, dedup tracking) |
| `hermes_calibration.db` | Calibration samples + settlement cross-validation |
| `last_run.txt` | Health check timestamp |

**Backup:**

```bash
# Simple copy (scheduler writes are atomic via SQLite WAL)
cp -r data/ data-backup-$(date +%Y%m%d)/

# Or tar it
tar czf hermes-data-$(date +%Y%m%d).tar.gz data/
```

The `data/` directory is mounted as a Docker volume — container rebuilds and
restarts do not lose data.

## Architecture

The container runs `python -m scheduler.cron --mode singapore --interval-minutes 30`:

1. Every 30 minutes: fetch Singapore temperature markets from Polymarket,
   get weather data, evaluate with DeepSeek LLM (5 samples per market),
   log decisions. Dedup ensures each market is evaluated at most once per day.
2. After each collection: check for newly settled markets, cross-validate
   Singapore outcomes against METAR data (IEM ASOS / WSSS).
3. Write `last_run.txt` with timestamp after each cycle.

**This is dry-run only** — no real orders are placed. The system produces
trading suggestions that are logged for calibration analysis.

## Scheduler Modes

| Mode | Markets | Interval |
|---|---|---|
| `singapore` | Singapore only | 30 min recommended |
| `all` | All cities on Polymarket | 8h recommended (rate limits) |

Override via CLI flags or env vars:

```bash
# CLI
python -m scheduler.cron --mode singapore --interval-minutes 30

# Env
SCHEDULER_MODE=singapore
SCHEDULER_INTERVAL_MINUTES=30
```
