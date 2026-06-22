#!/bin/bash

# Create root directory
mkdir -p wcb-ot

# Create all directories
mkdir -p wcb-ot/configs
mkdir -p wcb-ot/data/raw/sipakmed
mkdir -p wcb-ot/data/processed
mkdir -p wcb-ot/data/features
mkdir -p wcb-ot/data/splits
mkdir -p wcb-ot/src/data
mkdir -p wcb-ot/src/ot
mkdir -p wcb-ot/src/models
mkdir -p wcb-ot/src/train
mkdir -p wcb-ot/src/evaluation
mkdir -p wcb-ot/src/utils
mkdir -p wcb-ot/scripts
mkdir -p wcb-ot/tests
mkdir -p wcb-ot/results/models
mkdir -p wcb-ot/results/synthetic
mkdir -p wcb-ot/results/figures
mkdir -p wcb-ot/results/tables
mkdir -p wcb-ot/results/logs
mkdir -p wcb-ot/manuscript/sections
mkdir -p wcb-ot/manuscript/figures

# Create empty __init__.py in src/ subdirectories and tests/
touch wcb-ot/src/data/__init__.py
touch wcb-ot/src/ot/__init__.py
touch wcb-ot/src/models/__init__.py
touch wcb-ot/src/train/__init__.py
touch wcb-ot/src/evaluation/__init__.py
touch wcb-ot/src/utils/__init__.py
touch wcb-ot/tests/__init__.py

# Function to create placeholder python file
create_placeholder() {
  filepath=$1
  filename=$(basename "$filepath")
  echo "\"\"\"" > "$filepath"
  echo "Placeholder for $filename" >> "$filepath"
  echo "\"\"\"" >> "$filepath"
  echo "" >> "$filepath"
  echo "if __name__ == \"__main__\":" >> "$filepath"
  echo "    pass" >> "$filepath"
}

# Create placeholder files
create_placeholder wcb-ot/src/data/download.py
create_placeholder wcb-ot/src/data/preprocess.py
create_placeholder wcb-ot/src/data/features.py
create_placeholder wcb-ot/src/data/split.py

create_placeholder wcb-ot/src/ot/sinkhorn.py
create_placeholder wcb-ot/src/ot/barycenter.py
create_placeholder wcb-ot/src/ot/cyto_cost.py
create_placeholder wcb-ot/src/ot/dynamical.py

create_placeholder wcb-ot/src/models/wcb_ot.py
create_placeholder wcb-ot/src/models/resnet18.py
create_placeholder wcb-ot/src/models/stylegan2.py
create_placeholder wcb-ot/src/models/progressive_gan.py
create_placeholder wcb-ot/src/models/cvae.py
create_placeholder wcb-ot/src/models/baselines_classical.py

create_placeholder wcb-ot/src/train/train_classifier.py
create_placeholder wcb-ot/src/train/train_gans.py
create_placeholder wcb-ot/src/train/train_cvae.py

create_placeholder wcb-ot/src/evaluation/metrics.py
create_placeholder wcb-ot/src/evaluation/stat_tests.py
create_placeholder wcb-ot/src/evaluation/theoretical.py

create_placeholder wcb-ot/src/utils/logger.py
create_placeholder wcb-ot/src/utils/seed.py
create_placeholder wcb-ot/src/utils/io.py

create_placeholder wcb-ot/scripts/run_eda.py
create_placeholder wcb-ot/scripts/generate_synthetic.py
create_placeholder wcb-ot/scripts/evaluate_all.py
create_placeholder wcb-ot/scripts/make_figures.py
create_placeholder wcb-ot/scripts/reproduce_all.py

create_placeholder wcb-ot/tests/test_sinkhorn.py
create_placeholder wcb-ot/tests/test_barycenter.py
create_placeholder wcb-ot/tests/test_cyto_cost.py
create_placeholder wcb-ot/tests/test_pipeline.py

# Create .gitignore
cat <<EOT > wcb-ot/.gitignore
__pycache__/
*.pyc
.venv/
data/raw/*
data/processed/*
data/features/*
results/models/*.pt
results/synthetic/*
.mlflow/
*.egg-info/
EOT

# Create Makefile
cat <<EOT > wcb-ot/Makefile
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
EOT

# Create README.md
cat <<EOT > wcb-ot/README.md
# WCB-OT
Wasserstein Cellular Barycenters via Entropic Optimal Transport for synthetic rare cervical malignancies.
EOT

