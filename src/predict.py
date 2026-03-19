import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import csv

from torch_geometric.data import HeteroData

from config import *
from data_loader import *
from model import HeteroGraphMLP

# ---------------- main ----------------
def main():
    # 2) Load embeddings
    print("[load] loading embeddings...")
    chem_ids, chem_mat = load_chemical_embeddings_csv(CHEM_EMB_FILE, key_col="ID", dtype=np.float32)
    dis_ids, dis_names, dis_mat = load_all_disease_embeddings(DIS_EMB_GLOB, key_col="diseaseId", dtype=np.float32)

    dis_id_to_name = dict(zip(dis_ids, dis_names))
    
    chem_key2idx = {k: i for i,k in enumerate(chem_ids)}
    dis_key2idx = {k: i for i,k in enumerate(dis_ids)}

    chem_mat = chem_mat.astype(np.float32)
    dis_mat  = dis_mat.astype(np.float32)

    chem_tensor = torch.from_numpy(chem_mat)    
    dis_tensor  = torch.from_numpy(dis_mat)    

    chem_tensor = F.normalize(chem_tensor, p=2, dim=1)
    dis_tensor  = F.normalize(dis_tensor, p=2, dim=1)

    # 3) Build HeteroData (node features only)
    data = HeteroData()
    data['chemical'].x = chem_tensor
    data['disease'].x = dis_tensor 
    num_chem = data['chemical'].x.size(0); num_dis = data['disease'].x.size(0)
    print(f"[data] Nodes: chemical={num_chem:,}, disease={num_dis:,}")


    # 7) Build model
    in_ch = {
        'chemical': data['chemical'].x.size(1),
        'disease': data['disease'].x.size(1)
    }

    model = HeteroGraphMLP(in_ch, hidden_channels=hidden_channels, num_layers=num_layers, dropout=dropout, MLP_num_layers=MLP_num_layers, MLP_dropout=MLP_dropout).to(DEVICE)
    model = torch.compile(model, mode="max-autotune-no-cudagraphs", fullgraph=False, dynamic=True)


    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    # ---------------- Prediction ----------------
    chem_tensor = data['chemical'].x.to(DEVICE)
    dis_tensor  = data['disease'].x.to(DEVICE)

    with torch.no_grad():
        empty_edge_index = torch.empty((2,0), dtype=torch.long).to(DEVICE)

        edge_index_dict = {
            ('chemical','to','disease'): empty_edge_index,
            ('disease','rev_to','chemical'): empty_edge_index
        }

        x_dict = {
            'chemical': chem_tensor,
            'disease': dis_tensor
        }

        out_node_emb = model(x_dict, edge_index_dict)

        chem_emb = out_node_emb['chemical']
        dis_emb  = out_node_emb['disease']

        chem_batch_size = 256
        dis_batch_size  = 512 

        output_file = "outputs/predictions.csv"

        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Chemical ID", "Disease ID", "Disease Name", "Probability"])

            for i in range(0, chem_emb.size(0), chem_batch_size):

                chem_batch = chem_emb[i:i+chem_batch_size]

                for j in range(0, dis_emb.size(0), dis_batch_size):

                    dis_batch = dis_emb[j:j+dis_batch_size]

                    # Efficient pairwise scoring
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
                            score = probs[c_idx, d_idx].item()
                            disease_name = dis_names[j + d_idx]
                            writer.writerow([chem_id, disease_id, disease_name, score])

    print("Saved predictions")

if __name__ == "__main__":
    main()
    sys.exit(0)
