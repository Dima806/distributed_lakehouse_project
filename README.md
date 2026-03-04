# Distributed Lakehouse Lab: Spark vs Ray

A portfolio-grade distributed computing project that builds a complete data lakehouse and benchmarks **Apache Spark** against **Ray** for large-scale data workloads. Runs entirely on **GitHub Codespaces** (2 CPU cores, ≤8 GB RAM) — no cluster required.

---

## What This Project Demonstrates

| Topic | Details |
|-------|---------|
| **Lakehouse architecture** | Bronze → Silver → Gold medallion pattern |
| **Delta Lake** | ACID transactions, schema evolution, time travel, MERGE upserts |
| **Columnar formats** | Parquet, ORC, Delta read/write/filter benchmarks |
| **Spark patterns** | Group-by aggregations, broadcast vs shuffle joins, window functions, partition tuning |
| **Ray patterns** | Task parallelism (`@ray.remote`), actor model (stateful workers), Monte Carlo simulation |
| **Benchmarking** | Apples-to-apples engine comparison (wall time + peak memory) |
| **Modern Python tooling** | uv, ruff, prek, pytest, hatchling, GitHub Actions CI |

---

## Quick Start

```bash
# 1. Install Java 17, sync dependencies, and install git hooks
make install

# 2. Generate 5M-row synthetic e-commerce dataset
make generate-data

# 3. Build the full lakehouse (Bronze → Silver → Gold)
make pipeline

# 4. Run Spark distributed job suite
make run-spark

# 5. Run Ray distributed job suite
make run-ray

# 6. Run all benchmarks and write results to data/benchmark_results.json
make benchmark

# 7. Run test suite
make test
```

---

## Project Structure

```
├── src/
│   ├── utils/
│   │   ├── data_gen.py          # Synthetic dataset generator (5–20M rows)
│   │   ├── spark_session.py     # Shared Spark session factory
│   │   └── config.py            # YAML → SimpleNamespace config loader
│   ├── lakehouse/
│   │   ├── bronze.py            # Raw ingestion layer (Parquet, partitioned)
│   │   ├── silver.py            # Delta Lake: ACID, schema evolution, upserts
│   │   ├── gold.py              # Aggregated analytics tables
│   │   └── time_travel.py       # Delta time travel demonstration
│   ├── spark_jobs/
│   │   ├── aggregations.py      # Group-by aggregations
│   │   ├── joins.py             # Broadcast vs shuffle join comparison
│   │   ├── window_functions.py  # Rolling revenue and rank functions
│   │   ├── partition_tuning.py  # Shuffle partition impact (2 / 4 / 8)
│   │   └── main.py              # Orchestrator — prints timing summary
│   ├── ray_jobs/
│   │   ├── feature_engineering.py  # Task pattern: per-region feature compute
│   │   ├── simulation.py           # Task pattern: Monte Carlo price paths
│   │   ├── model_scoring.py        # Actor pattern: stateful per-region scorer
│   │   └── main.py                 # Orchestrator — prints timing summary
│   └── benchmarks/
│       ├── format_benchmark.py  # Parquet vs ORC vs Delta benchmarks
│       ├── compute_benchmark.py # Spark vs Ray aggregation benchmark
│       └── compare.py           # Unified runner — tables + JSON output
├── tests/                       # pytest suite (session-scoped Spark fixture)
├── config/config.yaml           # All tuneable parameters
├── data/                        # Generated data (git-ignored)
├── lakehouse/                   # Bronze / Silver / Gold tables (git-ignored)
├── Makefile                     # All runnable targets
├── pyproject.toml               # Dependencies and tool config
└── prek.toml                    # Pre-commit hooks (ruff lint + format check)
```

---

## Dataset

All data is **synthetically generated** from a fixed random seed (42) — no external downloads or network access.

### Schema

**Events** (5M rows default):

| Column | Type | Description |
|--------|------|-------------|
| `event_id` | int64 | Unique event identifier |
| `user_id` | int32 | Foreign key → users |
| `event_timestamp` | timestamp[us] | Event time (2024, uniform random) |
| `event_type` | string | purchase / view / add_to_cart / remove_from_cart / checkout |
| `product_id` | int32 | Foreign key → products |
| `price` | float64 | Unit price ($1–$500) |
| `quantity` | int8 | Units purchased (1–10) |
| `region` | string | north / south / east / west / central |

**Users** (100K rows): `user_id`, `age`, `region`, `signup_date`

**Products** (10K rows): `product_id`, `category`, `base_price`, `weight_kg`

Raw Parquet files are written in 1M-row chunks to `data/raw/`. Use `--rows` to scale:

```bash
uv run python -m src.utils.data_gen --rows 20000000   # 20M for full run
```

---

## Lakehouse Architecture

### Bronze Layer — Raw Ingestion (`src/lakehouse/bronze.py`)

Reads raw Parquet and writes to `lakehouse/bronze/` with no transformation. Events are
partitioned by `region` to enable partition pruning in downstream layers.

```
data/raw/events_*.parquet  →  lakehouse/bronze/events/region=north/...
                                                      /region=south/...
                                                      ...
```

### Silver Layer — ACID Delta Tables (`src/lakehouse/silver.py`)

Applies data quality rules and writes to `lakehouse/silver/events` as a **Delta Lake table**,
demonstrating three key capabilities in sequence:

**1. Initial write (Delta version 0)**
- Drops rows with nulls on key columns (`event_id`, `user_id`, `event_timestamp`, `event_type`)
- Filters out `price ≤ 0` and `quantity ≤ 0`
- Casts columns to canonical types (timestamp[us], double, int)
- Writes partitioned Delta table with `overwriteSchema=true`

**2. Schema evolution (version 1)**
- Adds a nullable `discount` column via `overwriteSchema=true`
- Backward-compatible schema change — existing queries still work against version 0

**3. ACID upsert (version 2)**
- Simulates 1,000 late-arriving corrected records
- Uses `DeltaTable.merge()` (`MERGE INTO` semantics):
  - Matched rows → update price (×0.9 multiplier) and set discount value
  - Unmatched rows → insert

**Time travel** (`src/lakehouse/time_travel.py`) queries any historical version:

```python
# Read the table exactly as it was after the initial write
spark.read.format("delta").option("versionAsOf", 0).load(SILVER_DIR)

# Full audit trail
spark.sql(f"DESCRIBE HISTORY delta.`{SILVER_DIR}/events`").show()
```

### Gold Layer — Analytics Tables (`src/lakehouse/gold.py`)

Produces four analytics-ready Delta tables from the silver layer:

| Table | Logic | Pattern |
|-------|-------|---------|
| `revenue_by_region_type` | `GROUP BY region, event_type` → sum revenue, count events | Aggregation |
| `daily_event_counts` | `GROUP BY date, region` → count events per day | Date grouping |
| `top_products` | Join events → products, rank by revenue, limit 10 | Join + sort |
| `rolling_revenue` | 7-day rolling sum per region | Window function |

---

## Spark Job Patterns (`make run-spark`)

All jobs share a single `SparkSession` factory (`src/utils/spark_session.py`) configured for
`local[2]` with 2 GB driver and 2 GB executor memory and Delta Lake extensions enabled.

### Aggregations (`src/spark_jobs/aggregations.py`)

Group-by aggregation over 5M silver events computing count, sum revenue, avg price, and max
price per `(region, event_type)`. Measures wall time from plan construction through the shuffle
to the final materialized Parquet write — capturing the full cost of a production-style ETL step.

### Broadcast vs Shuffle Join (`src/spark_jobs/joins.py`)

The same join query (events ⋈ products on `product_id`) executed with two strategies:

| Strategy | Mechanism | When to use |
|----------|-----------|-------------|
| **Broadcast join** | Small table replicated to every partition; no shuffle | Dimension tables < a few hundred MB |
| **Shuffle join** | Both sides repartitioned by join key (symmetric) | Two large tables of similar size |

Broadcast join is typically **2–3× faster** here because the products table (10K rows) fits
entirely in executor memory.

### Window Functions (`src/spark_jobs/window_functions.py`)

Two window computations over daily aggregated revenue, partitioned by region:

- **7-day rolling revenue**: `Window.rowsBetween(-6, 0)` applied to `sum(daily_revenue)`
- **Daily revenue rank**: `rank()` over `Window.orderBy(desc("daily_revenue"))`

Demonstrates Spark SQL's declarative window API and its automatic handling of ordering and
frame boundaries without explicit loops.

### Partition Tuning (`src/spark_jobs/partition_tuning.py`)

Runs the same group-by three times with `spark.sql.shuffle.partitions` set to **2**, **4**,
and **8**, capturing elapsed time for each. On a 2-core machine:

- **2 partitions** — underutilises parallelism; tasks may hit memory limits
- **4 partitions** — matches physical core count; typically optimal
- **8 partitions** — adds scheduling overhead without extra parallelism

---

## Ray Job Patterns (`make run-ray`)

All jobs call `ray.init(num_cpus=2)`. The project is installed as an editable package so Ray
workers can import `src.*` without packaging the working directory.

### Feature Engineering — Task Pattern (`src/ray_jobs/feature_engineering.py`)

Demonstrates **embarrassingly parallel** computation with `@ray.remote` tasks. One task is
dispatched per region; each reads its Parquet chunk(s) and independently computes event count,
mean/std price, total revenue, and avg quantity.

```python
@ray.remote
def engineer_region_features(region: str, raw_dir: str) -> dict:
    files = [f for f in Path(raw_dir).glob("events_*.parquet")]
    df = pd.concat([pd.read_parquet(f) for f in files])
    sub = df[df["region"] == region]
    ...

futures = [engineer_region_features.remote(r, str(RAW_DIR.resolve())) for r in REGIONS]
results = ray.get(futures)   # blocks until all five tasks complete
```

Five tasks run concurrently across 2 CPUs — Ray schedules them automatically.

### Monte Carlo Simulation — Task Pattern (`src/ray_jobs/simulation.py`)

Geometric Brownian Motion price simulation across **500 products × 252 trading days**. Each
`@ray.remote` task is fully independent (no shared state), making this a textbook Ray workload:
high parallelism, zero coordination overhead.

```python
@ray.remote
def simulate_price_path(product_id: int, base_price: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    returns = rng.normal(mu, sigma, size=N_STEPS)       # GBM returns
    prices = base_price * np.exp(np.cumsum(returns))    # price path
    return {"final_price": prices[-1], "volatility": prices.std(), ...}
```

### Model Scoring — Actor Pattern (`src/ray_jobs/model_scoring.py`)

Demonstrates Ray's **actor model** for stateful distributed computation. One `ScoringActor`
per region holds region-specific model weights in memory across multiple scoring calls —
avoiding the cost of re-serializing the model on every request.

```python
@ray.remote
class ScoringActor:
    def __init__(self, region: str):
        # weights live in actor process memory — not re-sent per call
        self.coef_price = {"north": 0.3, ...}[region]
        self.coef_qty   = {"north": 1.5, ...}[region]
        self.bias       = {"north": 2.0, ...}[region]

    def score_batch(self, prices, quantities) -> list[float]:
        return [self.coef_price * p + self.coef_qty * q + self.bias
                for p, q in zip(prices, quantities)]

# Create one long-lived actor per region
actors = {r: ScoringActor.remote(r) for r in REGIONS}
scores = ray.get([actors[r].score_batch.remote(prices, qtys) for r in REGIONS])
```

---

## Benchmarks (`make benchmark`)

Results are written to `data/benchmark_results.json` after each run.

### Columnar Format Comparison (`src/benchmarks/format_benchmark.py`)

Writes the same 5M-row dataset in Parquet, ORC, and reads from the existing Delta silver
table. Measures write time, full scan, filtered scan (predicate pushdown on `region="north"`),
and compressed size on disk.

| Metric | Parquet | ORC | Delta |
|--------|---------|-----|-------|
| Write time | ~28s | ~11s | — (pre-built) |
| Full scan | ~0.7s | ~0.4s | ~2.2s |
| Filtered scan | ~0.6s | ~0.9s | ~1.5s |
| Size on disk | ~124 MB | ~92 MB | varies* |

*Delta size includes the transaction log and all historical versions accumulated during the
silver pipeline run.

**Key takeaways:**
- ORC writes faster and compresses ~25% better than Parquet on this schema
- Delta's overhead comes from version metadata, not the underlying Parquet files
- Delta's filtered scan benefits from file-level stats in the transaction log (data skipping)

### Spark vs Ray Compute Comparison (`src/benchmarks/compute_benchmark.py`)

The identical aggregation task — `GROUP BY region, event_type` producing count, sum revenue,
avg price — implemented in both engines:

| Engine | Approach | Typical time |
|--------|----------|-------------|
| **Spark** | DataFrame API + Catalyst optimizer + columnar shuffle | ~4s |
| **Ray** | pandas per Parquet chunk + task fan-out | ~25s |

**Key takeaway:** Spark's Catalyst optimizer, whole-stage code generation, and columnar
execution make it far superior for SQL-style analytics on structured data. Ray wins when tasks
involve custom Python logic, ML inference, or truly independent computation (simulation, feature
engineering) where Spark's planning overhead outweighs its execution advantages.

---

## Configuration

All parameters live in `config/config.yaml` and are loaded as a `SimpleNamespace` tree:

```python
from src.utils.config import cfg

cfg.spark.master          # "local[2]"
cfg.spark.executor_memory # "2g"
cfg.ray.num_cpus          # 2
cfg.data_gen.n_rows       # 5_000_000
cfg.data_gen.seed         # 42
cfg.gold.rolling_window_days  # 7
cfg.benchmarks.filter_region  # "north"
```

To run with 20M rows, change `n_rows` in `config.yaml` or pass `--rows` to the generator.

---

## Testing

```bash
make test        # run all 20 tests
make lint        # ruff check (E, F, I, UP rules)
make format      # ruff format (line-length = 79)
```

Tests use a **session-scoped Spark fixture** (`tests/conftest.py`) that starts the JVM once for
the entire test run, avoiding the ~10s startup penalty per test module.

| File | Tests | What is covered |
|------|-------|-----------------|
| `test_data_gen.py` | 6 | Row counts, schema, no nulls, reproducibility across seeds |
| `test_bronze.py` | 3 | Partition directory structure, schema preservation |
| `test_silver.py` | 5 | Delta write/read, time travel, schema evolution, MERGE upsert, row filtering |
| `test_gold.py` | 3 | Aggregation non-empty output, revenue > 0, top-N limit |
| `test_benchmarks.py` | 3 | Parquet/ORC round-trip, filtered scan returns fewer rows |

---

## CI / CD

### Lint + Test — runs on every push and pull request to `main`

`.github/workflows/ci.yml`:

1. Install Java 17 (Microsoft distribution — same build as Codespaces)
2. Install uv, run `uv sync`
3. `make lint` — ruff check
4. `make test` — pytest (20 tests, single JVM session)

### Benchmark — manual dispatch only

`.github/workflows/benchmark.yml` — triggered from **Actions → Benchmark → Run workflow**:

- Optional `rows` input (default: 5,000,000)
- Runs the full pipeline: data generation → Spark jobs → Ray jobs → benchmarks
- Uploads `data/benchmark_results.json` as a downloadable artifact

---

## Environment Constraints

| Constraint | Value | Reason |
|-----------|-------|--------|
| CPU cores | 2 | GitHub Codespaces free tier |
| Driver + executor memory | 2 + 2 GB | Leaves headroom for OS + Ray |
| Java version | 17 | Maximum supported by PySpark 3.5 |
| Python | 3.12 | Latest stable; `setuptools` added as `distutils` shim |
| Random seed | 42 | Fully reproducible dataset across machines |
| External network | None | All data synthetic — runs offline after `make install` |

---

## Makefile Reference

| Target | Depends on | Description |
|--------|-----------|-------------|
| `make install` | — | Install Java 17 (sdkman), `uv sync`, install prek git hooks |
| `make lint` | — | `ruff check src tests` |
| `make format` | — | `ruff format src tests` |
| `make test` | — | `pytest` (20 tests) |
| `make generate-data` | — | Write 5M-row dataset to `data/raw/` |
| `make pipeline` | — | Bronze → Silver → Gold ETL |
| `make run-spark` | `pipeline` | All Spark job patterns + timing summary |
| `make run-ray` | `pipeline` | All Ray job patterns + timing summary |
| `make benchmark` | `pipeline` | Format + compute benchmarks → JSON |

---

## Dependencies

| Package | Version | Role |
|---------|---------|------|
| `pyspark` | 3.5.x | Distributed SQL / DataFrame engine |
| `delta-spark` | 3.2.x | Delta Lake ACID layer on top of Spark |
| `ray[default]` | 2.20+ | Task and actor distributed runtime |
| `duckdb` | 1.0+ | Embedded analytical SQL |
| `pandas` | 2.0+ | In-process data manipulation (Ray tasks) |
| `pyarrow` | 15.0+ | Parquet / ORC I/O backend |
| `numpy` | 1.26+ | Numerical computation (GBM simulation, data gen) |
| `psutil` | 5.9+ | RSS memory profiling in benchmarks |
| `pyyaml` | 6.0+ | `config/config.yaml` parsing |
| `setuptools` | 69.0+ | `distutils` shim required by PySpark on Python 3.12 |
| `pytest` | 8.0+ | Test framework |
| `ruff` | 0.4+ | Linting and formatting (replaces flake8 + black) |
