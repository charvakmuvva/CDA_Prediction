import glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def load_chemical_embeddings_csv(csv_path, key_col="ID", dtype=np.float32):
    df = pd.read_csv(csv_path, sep='\t')
    assert key_col in df.columns, f"{key_col} not found in CSV"

    ids = df[key_col].astype(str).tolist()
    emb_cols = [c for c in df.columns if c != key_col]

    emb_matrix = df[emb_cols].values.astype(dtype)
    return ids, emb_matrix
    
# ---------------- helpers ----------------
def load_embeddings_parquet_to_numpy(parquet_path, key_col, name_col='Disease Name', dtype=np.float16):
    table = pq.read_table(parquet_path, memory_map=True)
    cols = table.column_names
    
    assert key_col in cols, f"{key_col} not in {parquet_path}"
    assert name_col in cols, f"{name_col} not in {parquet_path}"
    
    ids = table.column(key_col).to_pylist()
    names = table.column(name_col).to_pylist()
    
    emb_cols = sorted(
        [c for c in cols if c.startswith("Emb")],
        key=lambda x: int(x.replace("Emb", ""))
    )

    arrs = [table.column(c).to_numpy() for c in emb_cols]
    emb_matrix = np.stack(arrs, axis=1).astype(dtype, copy=False)

    return ids, names, emb_matrix

def load_all_disease_embeddings(glob_pattern, key_col="diseaseId", dtype=np.float16):
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError("No disease embedding files found")

    all_ids, all_names, mats = [], [], []

    for f in files:
        ids, names, mat = load_embeddings_parquet_to_numpy(
            f, key_col=key_col, dtype=dtype
        )
        all_ids.extend(ids)
        all_names.extend(names)
        mats.append(mat)

    return all_ids, all_names, np.vstack(mats).astype(dtype, copy=False)

