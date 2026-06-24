.PHONY: install install-viz train ui pathways confusion p2cp format clean

install:
	uv sync --extra dev

install-viz:
	uv sync --extra dev --extra viz

train:
	uv run python -m nnviz.train

ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

p2cp:
	uv run python -m nnviz.plot_p2cp

format:
	uv run ruff format
	uv run ruff check --fix

clean:
	rm -rf frames movies
