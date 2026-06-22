"""
Check environment for WCB-OT project.
Imports critical packages and prints their versions.
"""

import sys

def check_env():
    try:
        import torch
        print(f"torch: {torch.__version__}")
    except ImportError:
        print("torch: NOT INSTALLED")

    try:
        import ot
        print(f"POT (ot): {ot.__version__}")
    except ImportError:
        print("POT (ot): NOT INSTALLED")

    try:
        import cv2
        print(f"opencv (cv2): {cv2.__version__}")
    except ImportError:
        print("opencv (cv2): NOT INSTALLED")

    try:
        import sklearn
        print(f"scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("scikit-learn: NOT INSTALLED")

    try:
        import albumentations
        print(f"albumentations: {albumentations.__version__}")
    except ImportError:
        print("albumentations: NOT INSTALLED")

    try:
        import timm
        print(f"timm: {timm.__version__}")
    except ImportError:
        print("timm: NOT INSTALLED")

    try:
        import mlflow
        print(f"mlflow: {mlflow.__version__}")
    except ImportError:
        print("mlflow: NOT INSTALLED")

    try:
        import optuna
        print(f"optuna: {optuna.__version__}")
    except ImportError:
        print("optuna: NOT INSTALLED")

    try:
        import statsmodels
        print(f"statsmodels: {statsmodels.__version__}")
    except ImportError:
        print("statsmodels: NOT INSTALLED")

    try:
        import loguru
        # loguru doesn't have __version__ in some versions or usually used as logger
        # but modern versions do.
        from loguru import __version__ as loguru_version
        print(f"loguru: {loguru_version}")
    except ImportError:
        print("loguru: NOT INSTALLED")

if __name__ == "__main__":
    print(f"Python Version: {sys.version}")
    print("-" * 30)
    check_env()
