# CLAUDE.md — Distributed Lakehouse Lab: Implementation Plan

## Project Summary

Portfolio-grade distributed computing project comparing Spark and Ray for lakehouse workloads. Runs entirely on GitHub Codespaces with 2 CPU cores. Demonstrates Delta Lake, columnar formats (Parquet/ORC), ACID transactions, schema evolution, time travel, and benchmarking.

---

## Phase 1: Project Scaffold

**Goal:** Establish the repo structure, tooling, and environment so every subsequent phase has a clean foundation.

### Tasks

1. **`pyproject.toml`** — define dependencies with `uv`:
   - `pyspark`, `delta-spark`, `ray[default]`, `duckdb`
   - `pytest`, `ruff`

2. **`Makefile`** — targets matching PRD §12:
   - `install`, `lint`, `format` (`ruff format`), `test`
   - `run-spark`, `run-ray`, `benchmark`

3. **Directory skeleton:**
   ```
   data/
   lakehouse/
     bronze/
     silver/
     gold/
   src/
     spark_jobs/
     ray_jobs/
     benchmarks/
     lakehouse/
     utils/
   tests/
   ```

4. **`.devcontainer/devcontainer.json`** — codespaces config:
   - Java 17 (required by Spark)
   - Python 3.12
   - `postCreateCommand: make install`

5. **Git hooks via `prek`:**
   ```
   uv tool install prek && prek install
   ```
   Hook to run on commit: `ruff check` + `ruff format --check`

6. **`src/__init__.py`** and per-package `__init__.py` stubs.

---

## Phase 2: Synthetic Dataset Generator

**Goal:** Produce a realistic, reproducible dataset (5–20M rows) that fits in 2-CPU memory.

### Tasks

1. **`src/utils/data_gen.py`**
   - Generate `events` table: `event_id`, `user_id`, `event_timestamp`, `event_type`, `product_id`, `price`, `quantity`, `region`
   - Generate `users` and `products` dimension tables
   - Configurable row count (default 5M for dev, 20M for full run)
   - Output raw Parquet files to `data/raw/`

2. **`tests/test_data_gen.py`** — assert schema and row count.

---

## Phase 3: Bronze Layer (Raw Ingestion)

**Goal:** Ingest raw Parquet data into the lakehouse bronze zone with no transformation.

### Tasks

1. **`src/lakehouse/bronze.py`**
   - Spark session: `local[2]`, memory-capped configs
   - Read `data/raw/*.parquet`
   - Write to `lakehouse/bronze/` as Parquet, partitioned by `region`

2. **`tests/test_bronze.py`** — validate row counts and partition columns.

---

## Phase 4: Silver Layer (Delta Lake)

**Goal:** Clean, validate, and write ACID-transactional Delta tables to the silver zone.

### Tasks

1. **`src/lakehouse/silver.py`**
   - Read from bronze
   - Apply cleaning: drop nulls in key columns, enforce types
   - Write to `lakehouse/silver/events` as Delta table
   - Demonstrate:
     - **Schema enforcement** — reject unexpected columns
     - **Schema evolution** — add a new column (`discount`) via `mergeSchema`
     - **ACID upsert** — `MERGE INTO` for late-arriving data

2. **`src/lakehouse/time_travel.py`**
   - Query silver Delta table at version 0 vs current
   - Print diff to stdout

3. **`tests/test_silver.py`** — validate Delta features, time travel query.

---

## Phase 5: Gold Layer (Aggregations)

**Goal:** Produce analytics-ready aggregated tables in the gold zone.

### Tasks

1. **`src/lakehouse/gold.py`**
   - Revenue by `region` and `event_type` (group-by + sum)
   - Daily event counts (window function)
   - Top 10 products by revenue (join events → products)
   - Write results as Delta tables to `lakehouse/gold/`

2. **`tests/test_gold.py`** — validate aggregation schemas and non-empty outputs.

---

## Phase 6: Spark Jobs

**Goal:** Demonstrate core Spark distributed patterns on the synthetic dataset.

### Tasks

1. **`src/spark_jobs/aggregations.py`** — group-by aggregations, measure time
2. **`src/spark_jobs/joins.py`** — broadcast join (events × products), shuffle join comparison
3. **`src/spark_jobs/window_functions.py`** — rolling revenue, rank by region
4. **`src/spark_jobs/partition_tuning.py`** — compare `spark.sql.shuffle.partitions` values
5. **`src/spark_jobs/main.py`** — orchestrate all jobs, print timing summary

Spark config used throughout:
```python
SparkSession.builder
    .master("local[2]")
    .config("spark.executor.memory", "2g")
    .config("spark.driver.memory", "2g")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
```

---

## Phase 7: Ray Jobs

**Goal:** Demonstrate Ray's task and actor model for parallel workloads.

### Tasks

1. **`src/ray_jobs/feature_engineering.py`** — parallel per-region feature transforms using `@ray.remote`
2. **`src/ray_jobs/simulation.py`** — parallel Monte Carlo price simulation (task pattern)
3. **`src/ray_jobs/model_scoring.py`** — distributed scoring with a dummy model (actor pattern)
4. **`src/ray_jobs/main.py`** — orchestrate all jobs, print timing summary

Ray init throughout:
```python
ray.init(num_cpus=2, ignore_reinit_error=True)
```

---

## Phase 8: Columnar Format Benchmark

**Goal:** Quantitatively compare Parquet, ORC, and Delta on read speed, filter speed, and storage size.

### Tasks

1. **`src/benchmarks/format_benchmark.py`**
   - Write the same 5M-row dataset in Parquet, ORC, Delta
   - Measure:
     - Full scan time
     - Filtered scan time (predicate pushdown on `region`)
     - File size on disk
     - Compression ratio
   - Output a comparison table

2. **`src/benchmarks/compare.py`** — unified entry point, calls format + compute benchmarks, writes results to `data/benchmark_results.json`

3. **`tests/test_benchmarks.py`** — smoke test that benchmark runs without error.

---

## Phase 9: Spark vs Ray Benchmark

**Goal:** Side-by-side timing comparison for equivalent workloads.

### Tasks

1. **`src/benchmarks/compute_benchmark.py`**
   - Run the same aggregation task in Spark and Ray
   - Capture wall time, peak memory (via `psutil`)
   - Append results to `data/benchmark_results.json`

2. Update `src/benchmarks/compare.py` to print a final markdown table:

   | Engine | Task           | Time (s) | Memory (MB) | Notes |
   |--------|----------------|----------|-------------|-------|
   | Spark  | Aggregation    | ...      | ...         |       |
   | Ray    | Aggregation    | ...      | ...         |       |

---

## Phase 10: Tests & CI Readiness

**Goal:** Full test coverage and a CI-ready Makefile.

### Tasks

1. All `tests/` files use `pytest` with fixtures for Spark session (session-scoped to avoid repeated JVM startup).
2. `conftest.py` — shared `spark` fixture at session scope.
3. `ruff` (lint) + `ruff format` enforce style.
4. `prek` enforces hooks on commit (`ruff check` + `ruff format --check`).
5. `make test` passes end-to-end on a cold codespace.

---

## Implementation Order

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
                                    ↓
Phase 6 (Spark) ←→ Phase 7 (Ray) → Phase 8 → Phase 9 → Phase 10
```

Phases 6 and 7 can be developed in parallel. Phase 8/9 depend on both.

---

## Constraints Checklist

- [ ] `SparkSession` always uses `local[2]`
- [ ] `ray.init` always uses `num_cpus=2`
- [ ] Driver + executor memory ≤ 6 GB total
- [ ] No external network calls (fully reproducible offline)
- [ ] All datasets generated synthetically from a fixed random seed
- [ ] `make test` is the single entry point for CI

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `pyproject.toml` | Dependency management (uv) |
| `Makefile` | All runnable targets |
| `.devcontainer/devcontainer.json` | Codespaces environment |
| `src/utils/data_gen.py` | Synthetic dataset generation |
| `src/lakehouse/bronze.py` | Raw ingestion layer |
| `src/lakehouse/silver.py` | Delta Lake / ACID layer |
| `src/lakehouse/gold.py` | Aggregated analytics layer |
| `src/lakehouse/time_travel.py` | Time travel demonstration |
| `src/spark_jobs/main.py` | Spark orchestration entry point |
| `src/ray_jobs/main.py` | Ray orchestration entry point |
| `src/benchmarks/compare.py` | Unified benchmark runner |
| `tests/conftest.py` | Shared pytest fixtures |
