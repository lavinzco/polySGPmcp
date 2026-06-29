# Hermes VPS Deployment

## First Deploy Checklist

按顺序执行，每一步确认通过再进入下一步。

```bash
# ── Step 1: 上传代码 ──
git clone <your-repo-url> hermes
cd hermes

# ── Step 2: 配置环境 ──
cp .env.example .env
nano .env   # 至少填 STRATEGY_API_KEY 和 CLASSIFICATION_API_KEY

# ── Step 3: 创建持久化目录 ──
mkdir -p data

# ── Step 4: 单独构建镜像（不要直接 up，先看 build 是否通过）──
docker-compose build
# 看到 "All imports OK" 说明所有 Python 依赖安装正确
# 如果这步失败，见下方「Build 失败排查」
```

**Build 通过后，用 `--once` 模式验证容器内的网络连通性和业务逻辑：**

```bash
# ── Step 5: 容器内单次执行（不启动长期循环）──
docker-compose run --rm hermes \
  python -m scheduler.cron --once --mode singapore

# 正常输出应包含：
#   "Fetching temperature events from Polymarket..."
#   "Found XXXX temperature markets"
#   "Filtered to XX markets for cities: {'Singapore'}"
#   "Evaluating XX markets for Singapore"
#   多条 DeepSeek API 200 OK
#   "Run complete: scanned=XX ..."
# 以及 data/ 目录下出现 hermes_decisions.db 和 last_run.txt

# 验证健康检查文件：
cat data/last_run.txt
# 应该显示 "2026-xx-xxTxx:xx:xx+00:00 ok"
```

**确认 Step 5 无报错后，正式启动长期循环：**

```bash
# ── Step 6: 正式启动 ──
docker-compose up -d
docker-compose logs -f   # 观察前几分钟确认正常
```

### Build 失败排查

| 现象 | 原因 | 解决 |
|---|---|---|
| `ModuleNotFoundError` in import check | pip install 漏装了某个包 | 检查 pyproject.toml dependencies 是否完整 |
| `error: subprocess-exited-with-error` during pip install | 某个包需要从源码编译但缺 C 编译器 | Dockerfile 里 pip install 前加一行：`RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*` |
| `Could not find a version that satisfies` | pip 版本太旧或 Python 版本不匹配 | 确认 `FROM python:3.11-slim`，Dockerfile 第一步已有 `pip install --upgrade pip` |
| 网络超时 `ConnectionError` | VPS 无法访问 pypi.org | 加 pip mirror：`RUN pip install --no-cache-dir -e . -i https://pypi.tuna.tsinghua.edu.cn/simple` |

> **当前所有生产依赖（httpx, pydantic, pydantic-settings, openai,
> anthropic, python-dotenv）都是纯 Python wheel，无 C 扩展，
> 在 slim 镜像上不需要 build-essential。** `siphon`（numpy/pandas/
> beautifulsoup4）只在 debug 脚本中使用，已被 .dockerignore 排除，
> 不会进入镜像。

### Step 5 失败排查

| 现象 | 原因 | 解决 |
|---|---|---|
| `httpx.ConnectError` to gamma-api.polymarket.com | VPS 网络或 DNS 问题 | `docker-compose run --rm hermes python -c "import httpx; print(httpx.get('https://gamma-api.polymarket.com/markets/1').status_code)"` |
| `httpx.ConnectError` to api.deepseek.com | DeepSeek API 被 VPS 地区屏蔽 | 换 VPS 地区，或在 .env 里换成其他 LLM provider |
| `httpx.ConnectError` to mesonet.agron.iastate.edu | IEM ASOS 不影响主循环，仅影响结算交叉验证 | 不阻塞部署，settlement check 会 log warning 后继续 |
| `httpx.ConnectError` to wttr.in | 天气数据源无法访问 | 换 VPS 地区，或考虑换天气 API |
| `KeyError` / `ValidationError` | .env 配置不完整 | 对照 .env.example 检查必填项 |

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
