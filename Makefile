SHELL := /bin/bash

.PHONY: setup

setup:
	@echo "Checking for uv..."
	@if ! command -v uv >/dev/null 2>&1; then \
		if [ -f "$$HOME/.local/bin/uv" ]; then \
			echo "uv found in $$HOME/.local/bin/uv"; \
			export PATH="$$HOME/.local/bin:$$PATH"; \
		else \
			echo "uv not found. Installing..."; \
			curl -LsSf https://astral.sh/uv/install.sh | sh; \
		fi \
	else \
		echo "uv is already installed."; \
	fi
	@echo "Ensuring .venv exists..."
	@export PATH="$$HOME/.local/bin:$$PATH"; \
	uv venv .venv --allow-existing
	@echo "Installing dependencies..."
	@export PATH="$$HOME/.local/bin:$$PATH"; \
	uv sync
