from src.utils.io import load_yaml
from pathlib import Path

def validate():
    configs = ["configs/config.yaml", "configs/ot_config.yaml", "configs/training_config.yaml"]
    for cfg in configs:
        data = load_yaml(Path(cfg))
        print(f"Successfully loaded {cfg}")
        if cfg == "configs/config.yaml":
            print(f"  Project: {data['project']['name']}")

if __name__ == "__main__":
    validate()
