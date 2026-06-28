# Industrial Vision Platform — common tasks
# Run `make help` to list targets.

.PHONY: help install install-core init-db demo test lint format api app train-anomaly evaluate clean

help:
	@echo "Available targets:"
	@echo "  install        Install full dependencies (incl. PyTorch)"
	@echo "  install-core   Install minimal deps (numpy/Pillow only, no torch)"
	@echo "  init-db        Create the SQLite traceability database"
	@echo "  demo           Run the end-to-end pipeline on sample images"
	@echo "  test           Run the test suite (pytest)"
	@echo "  lint           Check code style with ruff"
	@echo "  format         Auto-format code with ruff"
	@echo "  api            Start the FastAPI service on :8000"
	@echo "  app            Start the Streamlit app on :8501"
	@echo "  train-anomaly  Train the autoencoder anomaly detector"
	@echo "  evaluate       Evaluate a trained model on the test split"
	@echo "  clean          Remove caches and generated artifacts"

install:
	pip install -r requirements.txt

install-core:
	pip install -r requirements-core.txt

init-db:
	python -c "from src.database.db import init_db; init_db(); print('Database ready.')"

demo:
	python scripts/run_demo.py

test:
	pytest -v

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

app:
	streamlit run src/app/streamlit_app.py

train-anomaly:
	python -m src.training.prepare_data
	python -m src.training.train_anomaly

evaluate:
	python -m src.training.evaluate

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
	@echo "Cleaned caches (data/, models/, *.db left untouched)."
