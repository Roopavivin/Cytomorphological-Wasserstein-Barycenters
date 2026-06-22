install:
	pip install -r requirements.txt

test:
	pytest tests/

eda:
	python scripts/run_eda.py

train:
	python scripts/reproduce_all.py

evaluate:
	python scripts/evaluate_all.py

figures:
	python scripts/make_figures.py

reproduce:
	python scripts/reproduce_all.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
