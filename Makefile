.PHONY: install lint format test run-spark run-ray benchmark generate-data pipeline

SDKMAN_INIT := /usr/local/sdkman/bin/sdkman-init.sh
JAVA_17_HOME := /usr/local/sdkman/candidates/java/17.0.18-ms
# Allow CI (setup-java) to override JAVA_HOME; fall back to sdkman on Codespaces
JAVA_HOME ?= $(JAVA_17_HOME)
export JAVA_HOME

install:
	@if [ -f "$(SDKMAN_INIT)" ]; then \
		bash -c 'source $(SDKMAN_INIT) && sdk install java 17.0.18-ms || true'; \
	fi
	uv sync
	uv tool install prek && prek install

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

test:
	uv run pytest

generate-data:
	uv run python -m src.utils.data_gen

pipeline:
	uv run python -m src.lakehouse.bronze
	uv run python -m src.lakehouse.silver
	uv run python -m src.lakehouse.gold

run-spark: pipeline
	uv run python -m src.spark_jobs.main

run-ray: pipeline
	uv run python -m src.ray_jobs.main

benchmark: pipeline
	uv run python -m src.benchmarks.compare
