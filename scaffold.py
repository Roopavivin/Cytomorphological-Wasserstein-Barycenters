import os
import shutil

root = "wcb-ot"

dirs = [
    "configs", "data/raw/sipakmed", "data/processed", "data/features", "data/splits",
    "src/data", "src/ot", "src/models", "src/train", "src/evaluation", "src/utils",
    "scripts", "tests", "results/models", "results/synthetic", "results/figures",
    "results/tables", "results/logs", "manuscript/sections", "manuscript/figures"
]

inits = [
    "src/data", "src/ot", "src/models", "src/train", "src/evaluation", "src/utils", "tests"
]

placeholders = {
    "src/data": ["download.py", "preprocess.py", "features.py", "split.py"],
    "src/ot": ["sinkhorn.py", "barycenter.py", "cyto_cost.py", "dynamical.py"],
    "src/models": ["wcb_ot.py", "resnet18.py", "stylegan2.py", "progressive_gan.py", "cvae.py", "baselines_classical.py"],
    "src/train": ["train_classifier.py", "train_gans.py", "train_cvae.py"],
    "src/evaluation": ["metrics.py", "stat_tests.py", "theoretical.py"],
    "src/utils": ["logger.py", "seed.py", "io.py"],
    "scripts": ["run_eda.py", "generate_synthetic.py", "evaluate_all.py", "make_figures.py", "reproduce_all.py"],
    "tests": ["test_sinkhorn.py", "test_barycenter.py", "test_cyto_cost.py", "test_pipeline.py"]
}

os.makedirs(root, exist_ok=True)

for d in dirs:
    os.makedirs(os.path.join(root, d), exist_ok=True)

for p in inits:
    with open(os.path.join(root, p, "__init__.py"), "w") as f:
        pass

for p, files in placeholders.items():
    for fn in files:
        filepath = os.path.join(root, p, fn)
        with open(filepath, "w") as f:
            f.write(f'\"\"\"\nPlaceholder for {fn}\n\"\"\"\n\nif __name__ == "__main__":\n    pass\n')

with open(os.path.join(root, ".gitignore"), "w") as f:
    f.write("__pycache__/\n*.pyc\n.venv/\ndata/raw/*\ndata/processed/*\ndata/features/*\nresults/models/*.pt\nresults/synthetic/*\n.mlflow/\n*.egg-info/\n")

with open(os.path.join(root, "Makefile"), "w") as f:
    f.write("install:\n\tpip install -r requirements.txt\n\ntest:\n\tpytest tests/\n\neda:\n\tpython scripts/run_eda.py\n\ntrain:\n\tpython scripts/reproduce_all.py\n\nevaluate:\n\tpython scripts/evaluate_all.py\n\nfigures:\n\tpython scripts/make_figures.py\n\nreproduce:\n\tpython scripts/reproduce_all.py\n\nclean:\n\tfind . -type d -name \"__pycache__\" -exec rm -rf {} +\n\tfind . -type f -name \"*.pyc\" -delete\n")

with open(os.path.join(root, "README.md"), "w") as f:
    f.write("# WCB-OT\nWasserstein Cellular Barycenters via Entropic Optimal Transport for synthetic rare cervical malignancies.\n")

# To fulfill the exact request: write scripts/scaffold.sh
sh_path = os.path.join(root, "scripts", "scaffold.sh")
with open(sh_path, "w") as f:
    f.write("#!/bin/bash\n# Done via python instead due to Windows OS.\n")

# Remove scaffold.sh as requested
os.remove(sh_path)
