import yaml
import os
import torch
from types import SimpleNamespace

config_path = os.path.join(os.getcwd(), "config.yaml")

with open(config_path, "r") as f:
    _cfg = yaml.safe_load(f)

def _to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    return value

cfg = _to_namespace(_cfg)

cfg.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
cfg.num_workers = min(8, max(1, (os.cpu_count() or 1) // 2))
