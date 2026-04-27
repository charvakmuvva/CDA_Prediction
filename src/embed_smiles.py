import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd

from config import INPUT_SMILES_FILE, CHEM_EMB_FILE


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ChemBERTa embeddings from SMILES")

    parser.add_argument("--input", default=INPUT_SMILES_FILE,
                        help="Input CSV file. Default: ./data/input_smiles.csv")
    parser.add_argument("--output", default=CHEM_EMB_FILE,
                        help="Output TSV embedding file. Default: ./outputs/chemberta3_embeddings.tsv")
    parser.add_argument("--id_col", default="ID",
                        help="Chemical ID column name. Default: ID")
    parser.add_argument("--smiles_col", default="smiles",
                        help="SMILES column name. Default: smiles")
    parser.add_argument("--model_name", default="seyonec/ChemBERTa-zinc-base-v1",
                        help="HuggingFace ChemBERTa model name")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size. Default: 32")

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Input CSV: {args.input}")
    print(f"Output TSV: {args.output}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    with open(args.output, "w", encoding="utf-8") as out:
        header_written = False

        for chunk in pd.read_csv(args.input, chunksize=args.batch_size, dtype=str):

            if args.id_col not in chunk.columns:
                raise ValueError(f"Column '{args.id_col}' not found in input file")

            if args.smiles_col not in chunk.columns:
                raise ValueError(f"Column '{args.smiles_col}' not found in input file")

            smiles_list = chunk[args.smiles_col].fillna("").tolist()
            ids_list = chunk[args.id_col].tolist()

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

            final_df = pd.concat([pd.Series(ids_list, name=args.id_col), emb_df], axis=1)

            if not header_written:
                final_df.to_csv(out, sep="\t", index=False)
                header_written = True
            else:
                final_df.to_csv(out, sep="\t", index=False, header=False)

    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
