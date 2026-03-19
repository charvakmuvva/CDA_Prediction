import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd

from config import INPUT_SMILES_FILE, CHEM_EMB_FILE

model_name = "seyonec/ChemBERTa-zinc-base-v1"
batch_size = 32

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    out = open(CHEM_EMB_FILE, "w", encoding="utf-8")
    header_written = False

    for chunk in pd.read_csv(INPUT_SMILES_FILE, chunksize=batch_size, dtype=str):

        smiles_list = chunk["smiles"].fillna("").tolist()
        ids_list = chunk["ID"].tolist()

        with torch.no_grad():
            tokens = tokenizer(
                smiles_list,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)

            outputs = model(**tokens)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        emb_df = pd.DataFrame(embeddings)
        emb_df.columns = [f"Embed_{i:03d}" for i in range(1, emb_df.shape[1] + 1)]

        final_df = pd.concat([pd.Series(ids_list, name="ID"), emb_df], axis=1)

        if not header_written:
            final_df.to_csv(out, sep="\t", index=False)
            header_written = True
        else:
            final_df.to_csv(out, sep="\t", index=False, header=False)

    out.close()
    print(f"Saved → {CHEM_EMB_FILE}")

if __name__ == "__main__":
    main()
