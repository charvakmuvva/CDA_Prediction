import torch

# =========================
# PATHS
# =========================
INPUT_SMILES_FILE = "./data/input_smiles.csv"
CHEM_EMB_FILE = "./outputs/chemberta3_embeddings.tsv"
DIS_EMB_GLOB = "./Disease_embeddings/SVD_Disease_embeddings.parquet"
MODEL_PATH = "./models/best_model.pth"

# =========================
# MODEL SETTINGS
# =========================
hidden_channels = 256
num_layers = 3
dropout = 0.0
MLP_num_layers = 2
MLP_dropout = 0.0

# =========================
# DEVICE
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
