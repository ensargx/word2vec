import yaml
import os
import torch

config_path = os.path.join(os.getcwd(), "config.yaml")

with open(config_path, "r") as f:
    _cfg = yaml.safe_load(f)

EMB_DIM = _cfg['training']['emb_dim']
BATCH_SIZE = _cfg['training']['batch_size']
K_NEG = _cfg['training']['k_neg']
LEARNING_RATE = _cfg['training']['learning_rate']
EPOCHS = _cfg['training']['epochs']

CHECKPOINT_DIR = _cfg['paths']['checkpoint_dir']
LOG_FILE = _cfg['paths']['log_file']
DATASET_NAME = _cfg['paths']['dataset_name']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
