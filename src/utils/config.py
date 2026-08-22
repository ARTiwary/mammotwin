"""
Central config loader for MammoTwin.

Every script (preprocessing, training, inference, the FastAPI backend)
should load its settings through `load_config()` rather than hardcoding
paths or hyperparameters, so training and inference can never drift apart.
"""

import os
import random
import yaml

# Repo root = two levels up from this file (src/utils/config.py -> repo root)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, "config", "config.yaml")


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load config.yaml and resolve all relative paths to absolute paths."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    for key, rel_path in config["paths"].items():
        config["paths"][key] = os.path.join(REPO_ROOT, rel_path)

    return config


def set_global_seed(seed: int) -> None:
    """
    Set the random seed for every source of randomness we currently use.
    Extend this (torch, torch.cuda) once PyTorch is introduced in Phase 6.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


if __name__ == "__main__":
    cfg = load_config()
    set_global_seed(cfg["project"]["seed"])
    print("Loaded config for project:", cfg["project"]["name"])
    print("Seed set to:", cfg["project"]["seed"])
    print("Resolved paths:")
    for k, v in cfg["paths"].items():
        print(f"  {k}: {v}")