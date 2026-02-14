.PHONY: install format lint check run serve test help

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	uv sync --extra dev

format: ## Format code with ruff
	uv run ruff format .

lint: ## Lint code with ruff
	uv run ruff check .

check: format lint ## Format and lint code

run: ## Run the agent (CLI chat interface)
	uv run python -m src.main

serve: ## Start the FastAPI server (API + React frontend)
	uv run uvicorn src.server:app --reload --host 0.0.0.0 --port 8000

test: ## Run tests
	uv run pytest
