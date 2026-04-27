import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.data import HeteroData

from config import *
from data_loader import *
from model import HeteroGraphMLP


def parse_args():
    parser = argparse.ArgumentParser(description="Predict chemical-disease associations")

    parser.add_argument("--chem_emb", default=CHEM_EMB_FILE,
                        help="Chemical embedding TSV file")
    parser.add_argument("--disease_emb", default=DIS_EMB_GLOB,
                        help="Disease embedding parquet file or glob pattern")
    parser.add_argument("--model_path", default=MODEL_PATH,
                        help="Trained model checkpoint")
    parser.add_argument("--output", default="./outputs/predictions.csv",
                        help="Output prediction CSV file")

    parser.add_argument("--chem_id_col", default="ID",
                        help="Chemical ID column in chemical embedding file")
    parser.add_argument("--disease_id_col", default="diseaseId",
                        help="Disease ID column in disease embedding parquet")
    parser.add_argument("--chemical_batch_size", type=int, default=256,
                        help="Chemical batch size")
    parser.add_argument("--disease_batch_size", type=int, default=512,
                        help="Disease batch size")

    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile before loading model weights")

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("[load] loading embeddings...")
    print(f"Chemical embeddings: {args.chem_emb}")
    print(f"Disease embeddings:  {args.disease_emb}")
    print(f"Model:              {args.model_path}")
    print(f"Output:             {args.output}")

    chem_ids, chem_mat = load_chemical_embeddings_csv(
        args.chem_emb,
        key_col=args.chem_id_col,
        dtype=np.float32
    )

    dis_ids, dis_names, dis_mat = load_all_disease_embeddings(
        args.disease_emb,
        key_col=args.disease_id_col,
        dtype=np.float32
    )

    chem_mat = chem_mat.astype(np.float32)
    dis_mat = dis_mat.astype(np.float32)

    chem_tensor = torch.from_numpy(chem_mat)
    dis_tensor = torch.from_numpy(dis_mat)

    chem_tensor = F.normalize(chem_tensor, p=2, dim=1)
    dis_tensor = F.normalize(dis_tensor, p=2, dim=1)

    data = HeteroData()
    data["chemical"].x = chem_tensor
    data["disease"].x = dis_tensor

    print(f"[data] Nodes: chemical={chem_tensor.size(0):,}, disease={dis_tensor.size(0):,}")

    in_ch = {
        "chemical": data["chemical"].x.size(1),
        "disease": data["disease"].x.size(1)
    }

    model = HeteroGraphMLP(
        in_ch,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
        MLP_num_layers=MLP_num_layers,
        MLP_dropout=MLP_dropout
    ).to(DEVICE)

    if args.compile:
        model = torch.compile(
            model,
            mode="max-autotune-no-cudagraphs",
            fullgraph=False,
            dynamic=True
        )

    state = torch.load(args.model_path, map_location=DEVICE)

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        print("[warning] Direct loading failed. Trying to remove '_orig_mod.' prefix...")
        fixed_state = {}
        for k, v in state.items():
            if k.startswith("_orig_mod."):
                fixed_state[k.replace("_orig_mod.", "")] = v
            else:
                fixed_state[k] = v
        model.load_state_dict(fixed_state, strict=True)

    model.eval()

    chem_tensor = data["chemical"].x.to(DEVICE)
    dis_tensor = data["disease"].x.to(DEVICE)

    with torch.no_grad():
        empty_edge_index = torch.empty((2, 0), dtype=torch.long).to(DEVICE)

        edge_index_dict = {
            ("chemical", "to", "disease"): empty_edge_index,
            ("disease", "rev_to", "chemical"): empty_edge_index
        }

        x_dict = {
            "chemical": chem_tensor,
            "disease": dis_tensor
        }

        out_node_emb = model(x_dict, edge_index_dict)

        chem_emb = out_node_emb["chemical"]
        dis_emb = out_node_emb["disease"]

        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Chemical ID", "Disease ID", "Disease Name", "Probability"])

            for i in range(0, chem_emb.size(0), args.chemical_batch_size):
                chem_batch = chem_emb[i:i + args.chemical_batch_size]

                for j in range(0, dis_emb.size(0), args.disease_batch_size):
                    dis_batch = dis_emb[j:j + args.disease_batch_size]

                    c_expand = chem_batch.unsqueeze(1).repeat(1, dis_batch.size(0), 1)
                    d_expand = dis_batch.unsqueeze(0).repeat(chem_batch.size(0), 1, 1)

                    c_expand = c_expand.view(-1, chem_emb.size(1))
                    d_expand = d_expand.view(-1, dis_emb.size(1))

                    logits = model.decode_links(c_expand, d_expand)
                    probs = torch.sigmoid(logits.view(-1))
                    probs = probs.view(chem_batch.size(0), dis_batch.size(0))

                    for c_idx in range(chem_batch.size(0)):
                        chem_id = chem_ids[i + c_idx]

                        for d_idx in range(dis_batch.size(0)):
                            disease_id = dis_ids[j + d_idx]
                            disease_name = dis_names[j + d_idx]
                            score = probs[c_idx, d_idx].item()

                            writer.writerow([chem_id, disease_id, disease_name, score])

    print(f"Saved predictions → {args.output}")


if __name__ == "__main__":
    main()
    sys.exit(0)
