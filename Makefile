# Единые точки входа для разработки и CI.
# make check — то же самое, что выполняет CI.

PYTHON ?= .venv/bin/python
PIP    ?= .venv/bin/pip

.PHONY: help venv install lint fmt typecheck test coverage bench check clean

help: ## Показать список команд
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

venv: ## Создать виртуальное окружение
	python3 -m venv .venv
	$(PIP) install --upgrade pip

install: ## Установить пакет и dev-зависимости
	$(PIP) install -e ".[dev]"

lint: ## Линтер (ruff)
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

fmt: ## Форматировать код
	$(PYTHON) -m ruff format src tests
	$(PYTHON) -m ruff check --fix src tests

typecheck: ## Строгая проверка типов
	$(PYTHON) -m mypy

test: ## Тесты
	$(PYTHON) -m pytest

coverage: ## Тесты с покрытием
	$(PYTHON) -m pytest --cov=quantumlab --cov-report=term-missing

bench: ## Бенчмарки (полный набор появится в Этапе 4)
	$(PYTHON) -m pytest -m benchmark

check: lint typecheck test ## Всё вместе — как в CI

clean: ## Удалить артефакты
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
