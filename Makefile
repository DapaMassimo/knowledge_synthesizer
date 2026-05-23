.PHONY: install lint format type test test-unit test-int test-eval check run cli clean hooks

install:
	uv sync --all-groups

lint:
	uv run ruff check .
	uv run lint-imports

format:
	uv run ruff format .

type:
	uv run mypy src

test:
	uv run pytest --cov=knowledge_synthesizer --cov-report=term-missing --cov-fail-under=80

test-unit:
	uv run pytest tests/unit

test-int:
	uv run pytest tests/integration

test-eval:
	uv run pytest -m eval

check: lint type test

run:
	uv run streamlit run src/knowledge_synthesizer/entrypoints/streamlit_app.py

cli:
	uv run python -m knowledge_synthesizer.entrypoints.cli

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

hooks:
	uv run pre-commit install
